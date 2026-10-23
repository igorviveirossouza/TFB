"""Immutable-code snapshots and exact data copies for recovering B3 arrays.

This module uses only the standard library. It never modifies the source
checkout, original run_config.json, or original datasets. The original worker's
design-hash check must still pass before any recovered task is submitted.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile


CONTROL = '.b3_recovery'
IDENTITY = 'snapshot.json'
DATA_IDENTITY = 'dataset.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or str(path) == '.':
        raise ValueError(f'Unsafe snapshot path: {value!r}')
    return path


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n', encoding='utf-8')


@contextmanager
def _lock(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_name(destination.name + '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _extract_archive(archive_path, destination):
    """Extract regular git files; reject links, special files and traversal.

    Runtime data is excluded even if tracked by the historical commit. It is
    materialized separately, so TFB cannot auto-register unrelated CSV files.
    """
    files, seen = {}, set()
    with tarfile.open(archive_path, 'r:') as archive:
        for member in archive:
            relative = _relative(member.name)
            name = str(relative)
            if name in seen:
                raise ValueError(f'Duplicate archive path: {name}')
            seen.add(name)
            if not member.isdir() and not member.isfile():
                raise ValueError(f'Archive links/special entries are not supported: {name}')
            if relative.parts[0] == CONTROL:
                raise ValueError(f'Commit uses the reserved recovery directory: {name}')
            if relative.parts[0] == 'dataset':
                continue
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
            mode = member.mode & 0o777
            target.chmod(mode)
            files[name] = {'sha256': sha256_file(target), 'mode': mode}
    return files


def verify_snapshot(snapshot, original_commit=None):
    """Verify original code and reject added files outside explicit manifests.

    Recovery support files inside .b3_recovery are owned and sealed by the
    recovery driver, separately from this original-code identity.
    """
    snapshot = Path(snapshot)
    if snapshot.is_symlink():
        raise ValueError('Snapshot directory must not be a symlink')
    snapshot = snapshot.resolve()
    identity = _json(snapshot / CONTROL / IDENTITY)
    if identity.get('format') != 'B3_CODE_SNAPSHOT_V1':
        raise ValueError('Unknown recovery snapshot format')
    if original_commit is not None and identity.get('original_commit') != original_commit:
        raise ValueError('Existing snapshot belongs to a different original commit')
    files = identity['files']
    for name, expected in files.items():
        relative = _relative(name)
        path = snapshot.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected['sha256']:
            raise ValueError(f'Original snapshot file changed or missing: {name}')
        if stat.S_IMODE(path.stat().st_mode) != expected['mode']:
            raise ValueError(f'Original snapshot file mode changed: {name}')
    data_manifest = snapshot / CONTROL / DATA_IDENTITY
    data_files = set(_json(data_manifest)['files']) if data_manifest.is_file() else set()
    for path in snapshot.rglob('*'):
        name = path.relative_to(snapshot).as_posix()
        if path.is_symlink():
            raise ValueError(f'Snapshot must not contain symlinks: {name}')
        if path.is_file() and name not in files and name not in data_files and path.relative_to(snapshot).parts[0] != CONTROL:
            raise ValueError(f'Unexpected file in original-code snapshot: {name}')
    return identity


def create_snapshot(repo_root, original_commit, destination):
    """Archive an exact commit into a new directory, or verify an existing one."""
    if not isinstance(original_commit, str) or not re.fullmatch('[0-9a-fA-F]{40}', original_commit):
        raise ValueError('original_commit must be an exact 40-character hexadecimal commit SHA')
    original_commit = original_commit.lower()
    repo_root = Path(repo_root).resolve()
    destination = Path(destination).absolute()
    if destination.is_symlink():
        raise ValueError('Snapshot destination must not be a symlink')
    destination = destination.resolve()
    if destination == repo_root or repo_root in destination.parents:
        raise ValueError('Snapshot destination must be outside the source checkout')
    actual = subprocess.check_output(
        ['git', '-C', str(repo_root), 'rev-parse', '--verify', original_commit + '^{commit}'],
        text=True,
    ).strip()
    if actual != original_commit:
        raise ValueError('original_commit does not identify that exact commit')
    with _lock(destination):
        if destination.exists():
            verify_snapshot(destination, original_commit)
            return destination
        with tempfile.TemporaryDirectory(prefix='.' + destination.name + '.', dir=destination.parent) as temporary:
            temporary = Path(temporary)
            archived = temporary / 'original.tar'
            subprocess.run(['git', '-C', str(repo_root), 'archive', '--format=tar',
                            '--output=' + str(archived), original_commit], check=True)
            staged = temporary / 'snapshot'
            staged.mkdir()
            files = _extract_archive(archived, staged)
            (staged / CONTROL).mkdir()
            _write_json(staged / CONTROL / IDENTITY, dict(
                format='B3_CODE_SNAPSHOT_V1', original_commit=original_commit,
                archive_sha256=sha256_file(archived), excluded_runtime_prefix='dataset/',
                files=files,
            ))
            staged.rename(destination)
    verify_snapshot(destination, original_commit)
    return destination


def verify_dataset(snapshot):
    """Verify the independently frozen input CSV, calendar and TFB index."""
    snapshot = Path(snapshot)
    if snapshot.is_symlink():
        raise ValueError('Snapshot directory must not be a symlink')
    snapshot = snapshot.resolve()
    identity = _json(snapshot / CONTROL / DATA_IDENTITY)
    if identity.get('format') != 'B3_DATA_SNAPSHOT_V1':
        raise ValueError('Unknown recovery dataset format')
    for name, expected in identity['files'].items():
        relative = _relative(name)
        if relative.parts[:2] != ('dataset', 'forecasting'):
            raise ValueError(f'Data manifest path is outside the forecasting directory: {name}')
        path = snapshot.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f'Frozen dataset file changed or missing: {name}')
    directory = snapshot / 'dataset'
    for path in directory.rglob('*'):
        name = path.relative_to(snapshot).as_posix()
        if path.is_symlink() or (path.is_file() and name not in identity['files']):
            raise ValueError(f'Unexpected frozen dataset file: {name}')
    return identity


def materialize_dataset(snapshot, metadata):
    """Copy exactly the old prepared inputs; never recalculate returns or labels.

    Returns a JSON-safe manifest. The original plan's absolute paths are left
    unchanged; the original worker checks them while the benchmark reads the
    byte-identical local copy by dataset basename.
    """
    snapshot = Path(snapshot)
    if snapshot.is_symlink():
        raise ValueError('Snapshot directory must not be a symlink')
    snapshot = snapshot.resolve()
    source = Path(metadata['dataset']).resolve()
    calendar = Path(metadata['calendar']).resolve()
    requested = dict(dataset=str(source), calendar=str(calendar),
                     dataset_sha256=metadata['dataset_sha256'],
                     calendar_sha256=metadata['calendar_sha256'])
    for path, key in ((source, 'dataset_sha256'), (calendar, 'calendar_sha256')):
        if sha256_file(path) != requested[key]:
            raise ValueError(f'Original prepared data no longer matches the old plan: {path}')
    with _lock(snapshot):
        verify_snapshot(snapshot)
        manifest_path = snapshot / CONTROL / DATA_IDENTITY
        if manifest_path.exists():
            existing = verify_dataset(snapshot)
            if existing['original'] != requested:
                raise ValueError('Snapshot already contains a different prepared dataset')
            return existing
        if (snapshot / 'dataset').exists():
            raise ValueError('Refusing to overwrite an untracked snapshot dataset directory')
        index_path = source.parent / 'FORECAST_META.csv'
        with index_path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            rows = [row for row in reader if row.get('file_name') == source.name]
        if len(rows) != 1:
            raise ValueError('Original FORECAST_META.csv must contain exactly one row for the prepared dataset')
        if rows[0].get('size') != 'user' or int(rows[0]['length']) != metadata['n_rows']:
            raise ValueError('FORECAST_META.csv disagrees with the old prepared data metadata')
        with tempfile.TemporaryDirectory(prefix='.recovery-data-', dir=snapshot.parent) as temporary:
            staged = Path(temporary) / 'dataset'
            forecasts = staged / 'forecasting'
            (forecasts / '.b3_new').mkdir(parents=True)
            destination = forecasts / source.name
            calendar_destination = forecasts / '.b3_new' / calendar.name
            shutil.copyfile(source, destination)
            shutil.copyfile(calendar, calendar_destination)
            if (sha256_file(destination) != requested['dataset_sha256']
                    or sha256_file(calendar_destination) != requested['calendar_sha256']):
                raise ValueError('Original prepared data changed during snapshot copying')
            with (forecasts / 'FORECAST_META.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerow(rows[0])
            files = {path.relative_to(staged.parent).as_posix(): sha256_file(path)
                     for path in staged.rglob('*') if path.is_file()}
            identity = dict(format='B3_DATA_SNAPSHOT_V1', original=requested, files=files,
                            dataset=str(snapshot / 'dataset/forecasting' / source.name),
                            calendar=str(snapshot / 'dataset/forecasting/.b3_new' / calendar.name))
            staged.rename(snapshot / 'dataset')
            _write_json(manifest_path, identity)
    verify_snapshot(snapshot)
    return verify_dataset(snapshot)
