"""Recovery freezes original code and only the old plan's exact input data."""
import csv
import io
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from b3_recovery_snapshot import (  # noqa: E402
    _extract_archive, create_snapshot, materialize_dataset, sha256_file,
    verify_dataset, verify_snapshot,
)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


@pytest.fixture
def original(tmp_path):
    repo = tmp_path / 'shared repo'
    repo.mkdir()
    git(repo, 'init', '-q')
    git(repo, 'config', 'user.email', 'test@example.invalid')
    git(repo, 'config', 'user.name', 'Test')
    (repo / 'scripts').mkdir()
    (repo / 'scripts/worker.py').write_text('ORIGINAL = 1\n')
    (repo / 'dataset/forecasting').mkdir(parents=True)
    (repo / 'dataset/forecasting/unrelated.csv').write_text('old tracked data\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-qm', 'original')
    commit = git(repo, 'rev-parse', 'HEAD')
    return repo, commit, tmp_path / 'frozen snapshot'


@pytest.fixture
def prepared(original):
    repo, _, _ = original
    directory = repo / 'dataset/forecasting'
    dataset = directory / 'prepared.csv'
    dataset.write_text('date,value,cols\n2020-01-01,0.1,A\n')
    calendar = directory / '.b3_new/calendar.csv'
    calendar.parent.mkdir()
    calendar.write_text('date\n2020-01-01\n')
    with (directory / 'FORECAST_META.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file_name', 'size', 'length', 'freq'])
        writer.writeheader()
        writer.writerow(dict(file_name='prepared.csv', size='user', length=1, freq='D'))
        writer.writerow(dict(file_name='unrelated.csv', size='small', length=100, freq='D'))
    return dict(dataset=str(dataset), calendar=str(calendar), n_rows=1,
                dataset_sha256=sha256_file(dataset), calendar_sha256=sha256_file(calendar))


def test_exact_original_commit_ignores_dirty_shared_checkout(original):
    repo, commit, destination = original
    (repo / 'scripts/worker.py').write_text('CURRENT_MASTER = 2\n')
    (repo / 'new.py').write_text('untracked\n')
    before = git(repo, 'status', '--porcelain')
    snapshot = create_snapshot(repo, commit, destination)
    assert (snapshot / 'scripts/worker.py').read_text() == 'ORIGINAL = 1\n'
    assert not (snapshot / 'new.py').exists()
    assert not (snapshot / 'dataset').exists()
    identity = verify_snapshot(snapshot, commit)
    assert identity['excluded_runtime_prefix'] == 'dataset/'
    assert identity['original_commit'] == commit
    assert git(repo, 'status', '--porcelain') == before
    assert create_snapshot(repo, commit, destination) == destination


@pytest.mark.parametrize('commit', ['HEAD', 'abc123', '0' * 39, '--help', 'x' * 40])
def test_requires_full_commit_identity(original, commit):
    repo, _, destination = original
    with pytest.raises(ValueError, match='40-character'):
        create_snapshot(repo, commit, destination)
    assert not destination.exists()


def test_existing_unverified_directory_is_preserved(original):
    repo, commit, destination = original
    destination.mkdir()
    sentinel = destination / 'do-not-overwrite.txt'
    sentinel.write_text('keep me')
    with pytest.raises(FileNotFoundError):
        create_snapshot(repo, commit, destination)
    assert sentinel.read_text() == 'keep me'


@pytest.mark.parametrize('kind', ['change', 'missing', 'new', 'mode', 'link'])
def test_snapshot_detects_code_mutation_and_added_files(original, kind):
    repo, commit, destination = original
    create_snapshot(repo, commit, destination)
    worker = destination / 'scripts/worker.py'
    if kind == 'change':
        worker.write_text('ALTERED = 1\n')
    elif kind == 'missing':
        worker.unlink()
    elif kind == 'new':
        (destination / 'new_import.py').write_text('shadow module\n')
    elif kind == 'mode':
        worker.chmod(0o700)
    else:
        (destination / '.b3_recovery/link').symlink_to(repo)
    with pytest.raises(ValueError):
        verify_snapshot(destination, commit)
    with pytest.raises(ValueError):
        create_snapshot(repo, commit, destination)


def test_rejects_symlink_in_original_git_tree(original):
    repo, _, destination = original
    (repo / 'unsafe').symlink_to('/tmp')
    git(repo, 'add', 'unsafe')
    git(repo, 'commit', '-qm', 'link')
    commit = git(repo, 'rev-parse', 'HEAD')
    with pytest.raises(ValueError, match='links/special'):
        create_snapshot(repo, commit, destination)
    assert not destination.exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'directory/../../escape'])
def test_archive_path_traversal_rejected(tmp_path, name):
    archive = tmp_path / 'unsafe.tar'
    with tarfile.open(archive, 'w') as output:
        member = tarfile.TarInfo(name)
        member.size = 4
        output.addfile(member, io.BytesIO(b'data'))
    destination = tmp_path / 'target'
    destination.mkdir()
    with pytest.raises(ValueError, match='Unsafe snapshot path'):
        _extract_archive(archive, destination)
    assert not list(destination.iterdir())


def test_snapshot_must_be_outside_shared_repo(original):
    repo, commit, _ = original
    with pytest.raises(ValueError, match='outside'):
        create_snapshot(repo, commit, repo / 'recovery')


def test_materializes_only_exact_inputs_and_filtered_index(original, prepared):
    repo, commit, destination = original
    create_snapshot(repo, commit, destination)
    original_plan = dict(prepared)
    source_index = (repo / 'dataset/forecasting/FORECAST_META.csv').read_bytes()
    result = materialize_dataset(destination, prepared)
    assert prepared == original_plan
    assert result == verify_dataset(destination)
    assert result == materialize_dataset(destination, prepared)
    verify_snapshot(destination, commit)
    assert Path(result['dataset']).read_bytes() == Path(prepared['dataset']).read_bytes()
    assert Path(result['calendar']).read_bytes() == Path(prepared['calendar']).read_bytes()
    expected = {'dataset/forecasting/prepared.csv',
                'dataset/forecasting/.b3_new/calendar.csv',
                'dataset/forecasting/FORECAST_META.csv'}
    assert set(result['files']) == expected
    with (destination / 'dataset/forecasting/FORECAST_META.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [dict(file_name='prepared.csv', size='user', length='1', freq='D')]
    assert (repo / 'dataset/forecasting/FORECAST_META.csv').read_bytes() == source_index


@pytest.mark.parametrize('kind', ['csv', 'calendar', 'index', 'extra', 'symlink'])
def test_dataset_verification_detects_all_input_mutations(original, prepared, kind):
    repo, commit, destination = original
    create_snapshot(repo, commit, destination)
    manifest = materialize_dataset(destination, prepared)
    directory = destination / 'dataset/forecasting'
    if kind == 'csv':
        Path(manifest['dataset']).write_text('changed')
    elif kind == 'calendar':
        Path(manifest['calendar']).write_text('changed')
    elif kind == 'index':
        (directory / 'FORECAST_META.csv').write_text('changed')
    elif kind == 'extra':
        (directory / 'another.csv').write_text('unregistered data')
    else:
        (directory / 'another.csv').symlink_to(prepared['dataset'])
    with pytest.raises(ValueError):
        verify_dataset(destination)


def test_changed_original_data_is_not_copied(original, prepared):
    repo, commit, destination = original
    create_snapshot(repo, commit, destination)
    Path(prepared['dataset']).write_text('changed original')
    with pytest.raises(ValueError, match='no longer matches'):
        materialize_dataset(destination, prepared)
    assert not (destination / 'dataset').exists()


def test_changed_index_metadata_is_not_copied(original, prepared):
    repo, commit, destination = original
    create_snapshot(repo, commit, destination)
    prepared['n_rows'] = 999
    with pytest.raises(ValueError, match='disagrees'):
        materialize_dataset(destination, prepared)
    assert not (destination / 'dataset').exists()
