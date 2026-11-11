"""Robustness grid, real producer configuration and Slurm-free resume contracts."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('robust_refinement_helpers', ROOT/'tests/test_rankmix_refinement_launcher.py')
refinement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement)
helpers = refinement.helpers
WRAPPER = ROOT/'scripts/run_rankmix_robustez.sh'
SMALL = dict(TRAIN_VALUES='80', LOOKBACKS_VALUES='32', HORIZON_VALUES='1',
             SEEDS_VALUES='2026', RANK_MIX_WEIGHT_SPECS='.5 .5 0', RANK_MIX_MODES='global',
             LAMBDAS_VALUES='0 .5 1', NS_MASK_A_VALUES='.025', RANK_MIX_INCLUDE_ABLATIONS='0')


def environment(**overrides):
    env = helpers.environment()
    for key in list(env):
        if key.startswith(('B3_', 'SUPER_')) or key in {'CROSS_CONSTRUCTION', 'RANK_WINDOWS', 'RANK_WINDOWS_VALUES', 'TASK_HEAD_MODES'}:
            env.pop(key)
    env.update(overrides)
    return env


def run(*args, **overrides):
    return subprocess.run(['bash', str(WRAPPER), *args], env=environment(**overrides),
                          capture_output=True, text=True)


def fixture(path):
    env = helpers.fixture(path)
    shutil.copy(ROOT/'scripts/b3_experiment_env.sh', path/'scripts/b3_experiment_env.sh')
    (path/'scripts/b3_experiment.py').write_text('''import json, os, pathlib, sys
keys = ['OUT_ROOT','LOG_ROOT','RESULT_ROOT','RUN_TAG','GRES','GPU_PARTITION','GPU_TIME',
        'NODELIST','CPUS_PER_TASK','MEMORY','MAX_GPU_JOBS','B3_AFTER_JOB',
        'B3_CONFIG_SOURCE','B3_WORKER_LAUNCHER','MOMENTUM_MODE','MOMENTUM_INPUT_MODE']
row = dict(config=json.loads(os.environ['B3_INLINE_CONFIG']), argv=sys.argv,
           root=os.environ.get('OUT_ROOT'), **{k:os.environ.get(k) for k in keys})
path = pathlib.Path(os.environ['CAPTURE'])
with path.open('a') as stream: stream.write(json.dumps(row)+'\\n')
print(json.dumps(row))
if sys.argv[2] == 'submit':
    counter = path.with_suffix('.counter')
    job = int(counter.read_text())+1 if counter.exists() else 1001
    counter.write_text(str(job)); print(f'Job {job}: array local 0-0; tarefas globais 0-0')
''')
    return env


def captured(path, action=None):
    rows = helpers.records(path)
    return rows if action is None else [row for row in rows if row['argv'][2] == action]


class RobustnessLauncherTest(unittest.TestCase):
    def check_total(self, total, **overrides):
        r = run('--plan', **overrides)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f'TOTAL: {total} treinos unicos', r.stdout)
        return r

    def test_default_count_and_repetitions_are_explicit(self):
        r = self.check_total(10440)
        self.assertIn('Controles lambda=0: 72; mistos: 6912; ranking puro lambda=1: 3456', r.stdout)
        self.assertIn('cortes=60 70 80 90', r.stdout)
        self.assertIn('origin_anchored_windows', r.stdout)
        self.assertIn('GRES=vazio', r.stdout)
        self.check_total(3480, SEEDS_VALUES='2026')
        self.check_total(9288, RANK_MIX_INCLUDE_ABLATIONS='0')

    def test_lambda_aliases_and_cross_category_overlap_are_one_fit(self):
        settings = {**SMALL, 'LAMBDAS_VALUES': '0 .1 0.10',
                    'RANK_MIX_INCLUDE_ABLATIONS': '1',
                    'RANK_MIX_ABLATION_SPECS': '.5 .5 0',
                    'RANK_MIX_ABLATION_LAMBDAS': '0.100 0.1'}
        self.check_total(3, **settings)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            result = run('--dry-run', **fixture(path), **settings)
            self.assertEqual(result.returncode, 0, result.stderr)
            tasks = refinement.task_rows(captured(path))
            self.assertEqual(len(tasks), len({key for _, key in tasks}))

    def test_export_defaults_file_matches_launcher_and_derives_paths(self):
        reference = run('--print-exports')
        self.assertEqual(reference.returncode, 0, reference.stderr)
        command = '\n'.join(x for x in reference.stdout.splitlines() if x.startswith('export '))
        dump = '\npython3 -c \'import json,os; print(json.dumps(dict(os.environ)))\''
        expected = subprocess.run(['bash', '-c', command + dump], env=environment(),
                                  text=True, capture_output=True, check=True)
        actual = subprocess.run(['bash', '-c', 'source "$1"' + dump, 'bash',
                                  str(ROOT/'scripts/config_rankmix_robustez.defaults.sh')],
                                 env=environment(), text=True, capture_output=True, check=True)
        expected, actual = json.loads(expected.stdout), json.loads(actual.stdout)
        for line in command.splitlines():
            name = line.split('=', 1)[0].split()[1]
            self.assertEqual(actual[name], expected[name], name)

    def test_real_producer_three_models_four_cuts_and_matched_versions(self):
        settings = dict(MODEL='Nonstationary_Transformer iTransformer simpleFORMER',
                        TRAIN_VALUES='60 70 80 90', LOOKBACKS_VALUES='32 256 512', HORIZON_VALUES='1 25',
                        SEEDS_VALUES='2026', RANK_MIX_WEIGHT_SPECS='.5 .5 0;0 .75 .25',
                        RANK_MIX_MODES='global regional', LAMBDAS_VALUES='0 .5 1',
                        NS_MASK_A_VALUES='.01 .05', RANK_MIX_INCLUDE_ABLATIONS='0')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = fixture(path)
            r = run('--dry-run', **env, **settings)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = captured(path)
            self.assertEqual(len(rows), 25)  # one control + 2 vectors x 2 modes x 2 lambdas x 3 versions
            tasks = refinement.task_rows(rows)
            self.assertEqual(len(tasks), 1800)
            self.assertEqual(len({p for p, _ in tasks}), 1800)
            self.assertEqual(len({key for _, key in tasks}), 1800)
            self.assertEqual({key[0] for _, key in tasks}, set(settings['MODEL'].split()))
            self.assertEqual({key[1] for _, key in tasks}, {60, 70, 80, 90})
            self.assertEqual({key[2] for _, key in tasks}, {32, 256, 512})
            self.assertEqual({key[3] for _, key in tasks}, {1, 25})
            self.assertEqual(sum(key[6] == 0 for _, key in tasks), 72)
            plain = [key for _, key in tasks if key[5] == 'rankmix']
            prox = [key for _, key in tasks if key[5] == 'rankmix_proximity']
            self.assertEqual(len(plain), 576)
            self.assertEqual(len(prox), 1152)
            self.assertEqual({key[10] for key in prox}, {.01, .05})
            for row in rows:
                cfg = row['config']
                self.assertEqual(cfg['loss']['loss_cross_construction'], 'origin_anchored_windows')
                self.assertEqual(cfg['loss']['loss_rank_windows'], [1, 5, 10, 15, 20, 25])
                self.assertEqual(row['MOMENTUM_MODE'], 'off')
                self.assertEqual(row['MOMENTUM_INPUT_MODE'], 'off')
                self.assertIn('/protocolo=origin_anchored_windows/', row['root'])
                for defaults in cfg['model_defaults'].values():
                    self.assertIsNone(defaults['parallel_strategy'])
                    self.assertNotIn('task_head_mode', defaults)

    def test_blocks_accepts_multiple_k_and_does_not_use_origin_windows(self):
        settings = {**SMALL, 'HORIZON_VALUES':'1 5 25', 'K_LOSS_VALUES':'1 5 25',
                    'NS_RANKING_PROTOCOL':'blocks', 'RANK_WINDOWS_VALUES':'not-used'}
        self.check_total(27, **settings)  # three controls + six H/K pairs x 2 lambda x 2 versions
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = fixture(path)
            r = run('--dry-run', **env, **settings)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = captured(path); tasks = refinement.task_rows(rows)
            self.assertEqual(len(tasks), 27)
            self.assertEqual({(key[3], key[4]) for _, key in tasks if key[6] > 0},
                             {(1, 1), (5, 1), (5, 5), (25, 1), (25, 5), (25, 25)})
            for row in rows:
                self.assertEqual(row['config']['loss']['loss_cross_construction'], 'blocks')
                self.assertFalse(row['config']['loss'].get('loss_rank_windows'))
                self.assertIn('/protocolo=blocks/', row['root'])

    def test_ablations_and_temporal_losses_share_identical_controls(self):
        settings = {**SMALL, 'TEMPORAL_LOSSES_VALUES':'mse huber',
                    'RANK_MIX_INCLUDE_ABLATIONS':'1', 'RANK_MIX_ABLATION_SPECS':'.5 .5 0;1 0 0',
                    'RANK_MIX_ABLATION_LAMBDAS':'1'}
        # 2 controls + 4 mixed + 2 primary pure + 2 extra pure (overlap deduplicated).
        self.check_total(10, **settings)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = fixture(path)
            r = run('--dry-run', **env, **settings)
            self.assertEqual(r.returncode, 0, r.stderr)
            tasks = refinement.task_rows(captured(path))
            self.assertEqual(len(tasks), 10)
            self.assertEqual(len({key for _, key in tasks}), 10)
            self.assertEqual(sum(key[6] == 0 for _, key in tasks), 2)
            self.assertEqual(sum(key[6] == 1 for _, key in tasks), 4)

    def test_resources_paths_and_resume_marks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = fixture(path)
            overrides = dict(NS_PARTITION='medusas_dev', NS_GRES='gpu:1', NS_NODELIST='medusa6',
                             NS_MAX_CONCURRENT_JOBS='2', NS_TIME_LIMIT='01:20:00', NS_CPUS_PER_TASK='4',
                             NS_MEMORY='8G', NS_AFTER_JOB='12345', NS_LOG_ROOT=str(path/'custom logs'),
                             NS_RESULT_ROOT=str(path/'custom results'), NS_BATCH_SIZE='4')
            r = run('--train', **env, **SMALL, **overrides)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = captured(path, 'submit')
            self.assertEqual(len(rows), 5)
            self.assertEqual([row['B3_AFTER_JOB'] for row in rows], ['12345','1001','1002','1003','1004'])
            for row in rows:
                self.assertEqual(row['GRES'], 'gpu:1')
                self.assertEqual(row['GPU_PARTITION'], 'medusas_dev')
                self.assertEqual(row['GPU_TIME'], '01:20:00')
                self.assertEqual(row['NODELIST'], 'medusa6')
                self.assertEqual(row['CPUS_PER_TASK'], '4')
                self.assertEqual(row['MEMORY'], '8G')
                self.assertEqual(row['MAX_GPU_JOBS'], '2')
                self.assertTrue(row['LOG_ROOT'].startswith(str(path/'custom logs')))
                self.assertTrue(row['RESULT_ROOT'].startswith(str(path/'custom results')))
                self.assertEqual(row['config']['model_defaults']['iTransformer']['batch_size'], 4)
                self.assertEqual(row['B3_WORKER_LAUNCHER'], str(WRAPPER))
            retry = run('--train', **env, **SMALL, **overrides)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(len(captured(path, 'submit')), 5)
            changed = run('--train', **env, **SMALL, **{**overrides, 'NS_BATCH_SIZE':'8'})
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('Parametros alterados', changed.stderr)
            self.assertEqual(len(captured(path, 'submit')), 5)

    def test_parallel_branches_selection_and_frozen_worker(self):
        settings = dict(SMALL)
        branches = run('--list-branches', **settings)
        self.assertEqual(branches.returncode, 0, branches.stderr)
        selected = branches.stdout.splitlines()[-1].split(' | ')[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = fixture(path)
            r = run('--train', **env, **settings, NS_PARALLEL_BRANCHES='1', NS_AFTER_JOB='12345')
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(all(row['B3_AFTER_JOB'] == '12345' for row in captured(path, 'submit')))
            self.check_total(1, **settings, NS_ONLY_BRANCHES=selected)
            frozen = {'frozen': ['exact', 512], 'loss': {'loss_cross_construction':'origin_anchored_windows'}}
            r = run('--on-node', 'composto', 'worker', '--offset', '7', **env,
                    PYTHON_BIN=sys.executable, B3_INLINE_CONFIG=json.dumps(frozen),
                    B3_CONFIG_SOURCE='frozen-launcher', MODEL='invalid-after-submit',
                    NS_MODEL_DEFAULTS_JSON='broken-json', RANK_WINDOWS_VALUES='999',
                    LAMBDAS_VALUES='invalid-after-submit')
            self.assertEqual(r.returncode, 0, r.stderr)
            worker = captured(path, 'worker')[-1]
            self.assertEqual(worker['config'], frozen)
            self.assertEqual(worker['argv'][1:], ['composto','worker','--offset','7'])

    def test_print_exports_roundtrip_and_architecture_override(self):
        r = run('--print-exports', **SMALL)
        self.assertEqual(r.returncode, 0, r.stderr)
        exports = '\n'.join(line for line in r.stdout.splitlines() if line.startswith('export '))
        names = {line.split('=', 1)[0].split()[1] for line in exports.splitlines()}
        self.assertTrue({'MODEL','TRAIN_VALUES','NS_MODEL_DEFAULTS_JSON','NS_RANKING_PROTOCOL',
                         'RANK_WINDOWS_VALUES','NS_MASK_A_VALUES','NS_LOG_ROOT','NS_GRES','NS_RESULT_ROOT'} <= names)
        command = exports + '\n' + 'python3 -c \'import json,os; print(json.dumps(dict(os.environ)))\''
        replay = subprocess.run(['bash','-c',command], env=environment(), text=True, capture_output=True)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        exported = json.loads(replay.stdout)
        self.assertEqual(exported['NS_GRES'], '')
        self.assertEqual(exported['RANK_MIX_WEIGHT_SPECS'], '.5 .5 0')
        self.assertEqual(json.loads(exported['NS_MODEL_DEFAULTS_JSON'])['iTransformer']['d_model'], 128)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = fixture(path)
            defaults = '{"iTransformer":{"d_model":64,"d_ff":128,"n_heads":4,"norm":true}}'
            r = run('--dry-run', **env, **SMALL, NS_MODEL_DEFAULTS_JSON=defaults)
            self.assertEqual(r.returncode, 0, r.stderr)
            for row in captured(path):
                self.assertEqual(row['config']['model_defaults']['iTransformer']['d_model'], 64)
                self.assertEqual(row['config']['model_defaults']['iTransformer']['n_heads'], 4)

    def test_invalid_protocol_grid_and_architecture_fail_before_submission(self):
        invalid = [dict(NS_RANKING_PROTOCOL='unknown'), dict(K_LOSS_VALUES='5'),
                   dict(RANK_WINDOWS_VALUES='5 25'), dict(RANK_WINDOWS_VALUES='1 1'),
                   dict(RANK_WINDOWS_VALUES='0 1'), dict(NS_MASK_A_VALUES='0'),
                   dict(NS_MASK_A_VALUES='.025 .025'), dict(RANK_MIX_WEIGHT_SPECS='.5 .4 0'),
                   dict(RANK_MIX_MODES='unknown'), dict(RANK_MIX_FRACTION='.6'),
                   dict(LAMBDAS_VALUES='0 1.1'), dict(MODEL='missing'),
                   dict(NS_MODEL_DEFAULTS_JSON='[]'),
                   dict(NS_MODEL_DEFAULTS_JSON='{"iTransformer":{"task_head_mode":"separate"}}'),
                   dict(NS_MODEL_DEFAULTS_JSON='{"iTransformer":{"loss_cross_lambda":0.8}}'),
                   dict(MOMENTUM_MODE='residual_output'), dict(MOMENTUM_INPUT_MODE='feature'),
                   dict(RANK_MIX_INCLUDE_ABLATIONS='1', RANK_MIX_ABLATION_SPECS='')]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                r = run('--plan', **{**SMALL, **overrides})
                self.assertNotEqual(r.returncode, 0)


if __name__ == '__main__':
    unittest.main()
