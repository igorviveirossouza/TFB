"""Regression contracts for the isolated rank-target launcher and exporter."""
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import ordinal_experiment as legacy
from scripts import rank_target_experiment as rank


def settings(tmp_path=None, **updates):
    env = dict(MODEL='TSMixer', TRAIN_VALUES='80', LOOKBACKS_VALUES='4', HORIZON_VALUES='1 3',
        SEEDS_VALUES='2027', LAMBDAS_VALUES='0 0.5 1', NS_NUM_EPOCHS='1', NS_BATCH_SIZE='2',
        NS_TRAIN_RATIO_IN_TV='0.75', NS_MODEL_DEFAULTS_JSON='{}', NS_PYTHON_BIN=sys.executable,
        NS_VENV_PATH='', EXPERIMENT_TAG='tests/rank')
    if tmp_path is not None:
        env.update(PRED_ROOT=str(tmp_path/'pred'), NS_RESULT_ROOT=str(tmp_path/'result'),
                   NS_LOG_ROOT=str(tmp_path/'logs'), NS_RETURNS_FILE=str(tmp_path/'returns.csv'))
    env.update(updates)
    return rank.settings(env)


def test_new_defaults_and_legacy_settings_are_independent():
    before = legacy.settings({})
    new = rank.settings({})
    branches = rank.build_branches(new)
    after = legacy.settings({})
    assert before == after
    assert before['NS_CROSS_SCORE_NORMALIZATION'] == 'zscore'
    assert new['INPUT_KIND'] == 'rank' and new['INPUT_NORMALIZATION'] == 'none'
    assert new['MODEL'] == 'Nonstationary_Transformer'
    assert sum(len(branch['tasks']) for branch in branches) == 102
    assert all(branch['config']['target_kind'] == 'rank' for branch in branches)
    assert all(not values['norm'] for branch in branches for values in branch['config']['model_defaults'].values())


@pytest.mark.parametrize('family', ['ordinal', 'ranking'])
@pytest.mark.parametrize('input_kind', ['rank', 'log_return'])
def test_controls_and_pure_endpoints_are_not_duplicated(family, input_kind):
    s = settings(LOSS_FAMILY=family, INPUT_KIND=input_kind, TEMPORAL_LOSSES_VALUES='mse mae',
                 BUCKET_COUNTS_VALUES='3 5')
    branches = rank.build_branches(s)
    tasks = [t for branch in branches for t in branch['tasks']]
    # Two horizons, one control per temporal loss; lambda=1 only first temporal.
    assert sum(t['cross_lambda'] == 0 for t in tasks) == 4
    n_variants = 8 if family == 'ordinal' else 4
    assert sum(t['cross_lambda'] == 1 for t in tasks) == 2*n_variants
    assert sum(t['cross_lambda'] == .5 for t in tasks) == 4*n_variants
    assert all(branch['config']['input_kind'] == input_kind for branch in branches)


def test_evaluation_windows_do_not_change_training_configuration():
    a = rank.build_branches(settings(RANK_WINDOWS_VALUES='1'))
    b = rank.build_branches(settings(RANK_WINDOWS_VALUES='1 5 25'))
    assert [x['tasks'] for x in a] == [x['tasks'] for x in b]
    assert a == b
    assert a == rank.build_branches(settings(K_EVAL_VALUES='5 10'))
    assert all(branch['config']['loss']['loss_rank_windows'] == [] for branch in b)
    assert all(t['k'] in {None, 1} for branch in b for t in branch['tasks'])
    with pytest.raises(ValueError, match='somente avaliacao'):
        rank.build_branches(settings(NS_RANKING_PROTOCOL='origin_anchored_windows'))


def test_k_training_explicit_and_global_loss_parameters_are_consumed():
    s = settings(LOSS_FAMILY='ranking', CROSS_LOSSES_VALUES='ranknet listnet listmle approxndcg spearman hinge',
        K_LOSS_VALUES='3', HORIZON_VALUES='3', NS_RANKNET_ALPHA='2', NS_LISTNET_TAU='3',
        NS_LISTMLE_TAU='4', NS_APPROXNDCG_TAU='5', NS_SPEARMAN_TAU='6', NS_HINGE_MARGIN='.2')
    branches = rank.build_branches(s)
    expected = {'ranknet': ('loss_ranknet_alpha', 2), 'listnet': ('loss_listnet_tau', 3),
                'listmle': ('loss_listmle_tau', 4), 'approxndcg': ('loss_approxndcg_tau', 5),
                'spearman': ('loss_spearman_tau', 6), 'hinge': ('loss_hinge_margin', .2)}
    for branch in branches:
        for task in branch['tasks']:
            hyper = rank.task_hyper(branch['config'], task)
            assert hyper['seed'] == 2027
            if task['cross_lambda']:
                key, value = expected[task['loss']]
                assert hyper[key] == value
                assert hyper['loss_trade_window'] == 3
    with pytest.raises(ValueError, match='divida H'):
        rank.build_branches(settings(K_LOSS_VALUES='3', HORIZON_VALUES='1'))


def test_tsmixer_present_on_disk_resolves_without_changing_legacy_registry():
    label, spec = rank.resolve_rank_model('TSMixer')
    assert label == 'TSMixer'
    assert spec['model_name'] == 'time_series_library.models.TSMixer.TSMixer'
    assert spec['adapter'] == 'transformer_adapter'
    with pytest.raises(ValueError):
        rank.resolve_rank_model('NonexistentTFBModel')


def test_read_only_plan_and_shell(tmp_path):
    s = settings(tmp_path)
    env = dict(__import__('os').environ, **s)
    result = subprocess.run(['bash', 'scripts/run_rank_target_experiment.sh', '--trainplan'],
                            cwd=rank.ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'TOTAL:' in result.stdout and 'ranking diario' in result.stdout
    assert not Path(s['PRED_ROOT']).exists()
    assert not Path(s['NS_LOG_ROOT']).exists()


def test_array_chunks_use_local_indices_and_do_not_exceed_limit(tmp_path):
    s = settings(tmp_path, NS_ARRAY_CHUNK_SIZE='3', NS_MAX_CONCURRENT_JOBS='2')
    branches = rank.build_branches(s)
    commands = rank.array_commands(s, branches, tmp_path/'snapshot.json')
    assert any(command['start'] == 3 for command in commands)
    for command in commands:
        assert command['stop'] - command['start'] < 3
        assert f'--array=0-{command["stop"]-command["start"]}%2' in command['args']
    assert len({command['script'] for command in commands}) == len(commands)


@pytest.mark.parametrize('parallel', ['0', '1'])
def test_slurm_dependencies_are_frozen_and_repeated_submission_is_idempotent(tmp_path, monkeypatch, parallel):
    s = settings(tmp_path, NS_ARRAY_CHUNK_SIZE='3', NS_PARALLEL_BRANCHES=parallel)
    branches = rank.build_branches(s)
    snapshot = tmp_path/'design.json'
    monkeypatch.setattr(rank, 'prepare', lambda *_: (snapshot, {'design_sha256': 'fixed'}))
    monkeypatch.setattr(rank.shutil, 'which', lambda _: '/bin/sbatch')
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=str(100+len(calls))+';cluster\n', stderr='')
    monkeypatch.setattr(rank.subprocess, 'run', run)
    result = rank.submit(s, branches)
    assert result['status'] == 'submitted'
    assert len(calls) == len(result['arrays'])
    for i, row in enumerate(result['arrays']):
        script = Path(row['script']).read_text()
        assert f'+ {row["start"]}' in script
        assert str(snapshot) in script
        if parallel == '0':
            assert row['dependency'] == (str(100+i) if i else '')
        elif row['start'] == 0:
            assert row['dependency'] == ''
    count = len(calls)
    assert rank.submit(s, branches)['job_ids'] == result['job_ids']
    assert len(calls) == count


def write_returns(path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(60, 5))*.01, index=pd.date_range('2020-01-01', periods=60),
                         columns=['A', 'B', 'C', 'D', 'E'])
    frame.index.name = 'date'
    frame.to_csv(path)
    return frame


def test_frozen_inputs_export_rank_semantics_and_worker_resume(tmp_path, monkeypatch):
    import torch
    s = settings(tmp_path, HORIZON_VALUES='3', LAMBDAS_VALUES='0', NS_MODEL_DEFAULTS_JSON='{}')
    source = write_returns(Path(s['NS_RETURNS_FILE']))
    branches = rank.build_branches(s)
    calls = []
    class FakeModel:
        def __init__(self):
            self.model = torch.nn.Linear(5, 5)
            self.criterion = torch.nn.MSELoss()
            self.history_, self.fit_metadata_ = [], {'target_kind': 'rank'}
        def forecast_fit(self, frame, **kwargs):
            calls.append(('fit', len(frame)))
            assert len(frame) == 48
        def predict_origins(self, frame, origins):
            calls.append(('predict', origins.copy()))
            return np.tile([1.4, 4.5, 2.3, 5.6, 3.7], (len(origins), 3, 1))
    monkeypatch.setattr(rank, 'model_instance', lambda *_: FakeModel())
    monkeypatch.setattr(rank, 'code_hashes', lambda: {'test': 'fixed'})
    snapshot, design = rank.prepare(s, branches)
    assert design['metadata']['prediction_kind'] == 'rank_position'
    assert design['metadata']['source_data_kind'] == 'log_return'
    source.iloc[:, :] = 999
    source.to_csv(s['NS_RETURNS_FILE'])  # Worker must consume the frozen copy.
    result = rank.worker(snapshot, 0, 0)
    assert result['fit_metadata']['target_kind'] == 'rank'
    output = Path(s['PRED_ROOT'])/branches[0]['relative']/rank.b3.output_path(branches[0]['tasks'][0])
    pred = pd.read_parquet(output/'part.parquet')
    assert pred.columns.tolist() == ['arquivo', 'A', 'B', 'C', 'D', 'E', 'h', 'origin_step', 'step']
    assert pred.origin_step.tolist() == np.repeat(np.arange(48, 58), 3).tolist()
    assert pred.loc[0, ['A', 'B', 'C', 'D', 'E']].tolist() == [1.4, 4.5, 2.3, 5.6, 3.7]
    assert Path(result['checkpoint']).is_file()
    assert str(result['checkpoint']).startswith(s['NS_RESULT_ROOT'])
    n_calls = len(calls)
    rank.worker(snapshot, 0, 0)
    assert len(calls) == n_calls
    pred.loc[0, 'A'] = 17
    pred.to_parquet(output/'part.parquet')
    with pytest.raises(ValueError, match='concluida foi alterada'):
        rank.worker(snapshot, 0, 0)


def test_prepared_root_cannot_be_repurposed_for_other_configuration(tmp_path, monkeypatch):
    s = settings(tmp_path, HORIZON_VALUES='1', LAMBDAS_VALUES='0')
    write_returns(Path(s['NS_RETURNS_FILE']))
    monkeypatch.setattr(rank, 'model_instance', lambda *_: object())
    monkeypatch.setattr(rank, 'code_hashes', lambda: {'test': 'fixed'})
    snapshot, _ = rank.prepare(s, rank.build_branches(s))
    original = snapshot.read_bytes()
    changed = dict(s, INPUT_KIND='log_return')
    with pytest.raises(ValueError, match='Configuracao, dados ou codigo'):
        rank.prepare(changed, rank.build_branches(changed))
    assert snapshot.read_bytes() == original
