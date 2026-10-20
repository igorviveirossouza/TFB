"""I/O failures must retain the failing operation even without a filename."""
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('filename', [None, '/snfs2/example/part.parquet'])
def test_prepare_io_error_keeps_traceback(monkeypatch, capsys, filename):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from backtest_composto import cli

    def failing_read(args):
        raise OSError(70, 'Communication error on send', filename)

    monkeypatch.setattr(cli, 'prepare', failing_read)
    monkeypatch.setattr(sys, 'argv', ['backtest', 'prepare', '--predictions', 'predictions',
                                    '--market', 'market', '--returns-file', 'returns',
                                    '--strategies', 'strategies', '--run-dir', 'run', '--dry-run'])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 1
    error = capsys.readouterr().err
    assert 'Traceback (most recent call last)' in error
    assert 'failing_read' in error and '[Errno 70]' in error
    if filename:
        assert filename in error


def test_router_io_error_keeps_traceback(monkeypatch, capsys, tmp_path):
    contract = tmp_path/'contrato.json'
    original_exists = Path.exists

    def failing_stat(path):
        if path == contract:
            raise OSError(70, 'Communication error on send')
        return original_exists(path)

    monkeypatch.setattr(Path, 'exists', failing_stat)
    monkeypatch.setattr(sys, 'argv', ['backtest', 'prepare', '--run-dir', str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT/'scripts/run_backtests_composto.py'), run_name='__main__')
    assert result.value.code == 1
    error = capsys.readouterr().err
    assert 'Traceback (most recent call last)' in error
    assert 'failing_stat' in error and '[Errno 70]' in error
