"""Staged mixture/A experiments: routing and resumability without Slurm jobs."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rankmix_test_helpers', ROOT/'tests/test_rankmix_launcher.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
WRAPPER = ROOT/'scripts/run_ns_rankmix_refinamento.sh'


def run(*args, **overrides):
    return subprocess.run(['bash', str(WRAPPER), *args],
                          env=helpers.environment(**overrides), capture_output=True, text=True)


def task_rows(records):
    spec = importlib.util.spec_from_file_location('rankmix_refinement_b3', ROOT/'scripts/b3_experiment.py')
    b3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b3)
    out = []
    for row in records:
        for task in b3.make_tasks(row['config'], row['argv'][1]):
            _, hyper, _, _ = b3.model_spec(row['config'], task)
            key = (task['modelo'], task['treino'], task['lookback'], task['pred_len'], task['k'],
                   task['loss'], task['cross_lambda'],
                   row['config']['loss']['loss_temporal'] if task['cross_lambda'] < 1 else None,
                   row['config']['strategy']['seed'], task.get('mask_type'), task.get('mask_a'),
                   tuple(hyper.get('loss_rank_mix_weights', [])), hyper.get('loss_rank_mix_mode'))
            out.append((str(Path(row['root'])/b3.output_path(task)), key))
    return out


class RefinementLauncherTest(unittest.TestCase):
    def check_total(self, n, **env):
        r = run('--plan', **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f'TOTAL: {n} treinos unicos', r.stdout)
        return r

    def test_composition_counts_and_editable_axes(self):
        self.check_total(198)
        self.check_total(396, MODEL='Nonstationary_Transformer DUET')
        self.check_total(396, SEEDS_VALUES='2026 2027')
        self.check_total(348, TEMPORAL_LOSSES_VALUES='mse huber')
        self.check_total(66, LOOKBACKS_VALUES='256')
        self.check_total(9, LOOKBACKS_VALUES='32', HORIZON_VALUES='1',
                         LAMBDAS_VALUES='0 .5 1', RANK_MIX_WEIGHT_SPECS='.5 .5 0;0 1 0')
        self.assertNotEqual(run('--plan', HORIZON_VALUES='1 25', K_LOSS_VALUES='5').returncode, 0)

    def test_scale_requires_a_deliberate_selected_mixture(self):
        r = run('--plan', RANK_MIX_STAGE='escala')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('RANK_MIX_SELECTED_WEIGHTS', r.stderr)
        self.check_total(102, RANK_MIX_STAGE='escala', RANK_MIX_SELECTED_WEIGHTS='0 .75 .25')
        self.check_total(78, RANK_MIX_STAGE='escala', RANK_MIX_SELECTED_WEIGHTS='0 .75 .25',
                         NS_MASK_A_VALUES='.01 .025')

    def test_invalid_or_duplicate_design_is_rejected_before_submit(self):
        invalid = [dict(RANK_MIX_STAGE='invalid'),
                   dict(RANK_MIX_WEIGHT_SPECS='.5 .5 0;0.50 .50 0.0'),
                   dict(RANK_MIX_WEIGHT_SPECS='.5 .25 0'),
                   dict(MOMENTUM_INPUT_MODE='feature'),
                   dict(NS_ONLY_BRANCHES='does_not_exist')]
        for a in ['.025 0.025', '0', '-.1', 'nan', '1e999']:
            invalid.append(dict(RANK_MIX_STAGE='escala', RANK_MIX_SELECTED_WEIGHTS='.25 .75 0',
                                NS_MASK_A_VALUES=a))
        for env in invalid:
            with self.subTest(env=env):
                self.assertNotEqual(run('--plan', **env).returncode, 0)

    def test_real_producer_union_has198_then246_unique_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            env = helpers.fixture(p)
            for name in ['run_temporal_experiment.sh', 'run_composite_trading_experiment.sh', 'b3_experiment_env.sh']:
                shutil.copy(ROOT/'scripts'/name, p/'scripts'/name)
            (p/'scripts/b3_experiment.py').write_text(
                "import json,os,sys\nprint(json.dumps(dict(config=json.loads(os.environ['B3_INLINE_CONFIG']), root=os.environ['OUT_ROOT'], argv=sys.argv)))\n")
            stages = {}
            for stage, settings in [('composicao', {}), ('escala', {'RANK_MIX_SELECTED_WEIGHTS': '0 .75 .25'})]:
                r = run('--dry-run', **env, RANK_MIX_STAGE=stage, NS_GRES='', **settings)
                self.assertEqual(r.returncode, 0, r.stderr)
                stages[stage] = [json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
                for row in stages[stage]:
                    row['paths'] = [path for path, _ in task_rows([row])]
            first = task_rows(stages['composicao'])
            second = task_rows(stages['escala'])
            self.assertEqual(len(first), 198)
            self.assertEqual(len(second), 102)
            self.assertEqual(len({path for path, _ in first}), 198)
            self.assertEqual(len({key for _, key in first}), 198)
            self.assertEqual(len(set(first) | set(second)), 246)
            self.assertEqual(len({path for path, _ in first + second}), 246)
            self.assertEqual(len({key for _, key in first + second}), 246)
            for path, _ in first + second:
                self.assertLessEqual(sum(part.startswith('a=') for part in Path(path).parts), 1)

    def test_stage_two_reuses_successful_marks_and_does_not_depend_on_expired_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); env = helpers.fixture(p)
            first = run('--train', **env, NS_GRES='', NS_MAX_CONCURRENT_JOBS='3')
            self.assertEqual(first.returncode, 0, first.stderr)
            rows1 = [r for r in helpers.records(p) if r['action'] == 'submit']
            self.assertEqual(len(rows1), 17)
            second = run('--train', **env, RANK_MIX_STAGE='escala',
                         RANK_MIX_SELECTED_WEIGHTS='0 .75 .25', NS_GRES='', NS_MAX_CONCURRENT_JOBS='3')
            self.assertEqual(second.returncode, 0, second.stderr)
            rows2 = [r for r in helpers.records(p) if r['action'] == 'submit'][17:]
            self.assertEqual(len(rows2), 4)
            self.assertEqual([r['B3_AFTER_JOB'] for r in rows2], ['', '1018', '1019', '1020'])
            self.assertTrue(all(r['CROSS_LOSSES'] == 'rankmix_proximity' for r in rows2))
            self.assertTrue(all(r['GRES'] == '' and r['MAX_CONCURRENT_JOBS'] == '3' for r in rows2))
            retry = run('--train', **env, RANK_MIX_STAGE='escala', RANK_MIX_SELECTED_WEIGHTS='0 .75 .25')
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(len([r for r in helpers.records(p) if r['action'] == 'submit']), 21)
            changed = run('--train', **env, NS_BATCH_SIZE='8')
            self.assertNotEqual(changed.returncode, 0)

    def test_parallel_arrays_honor_explicit_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); env = helpers.fixture(p)
            r = run('--train', **env, NS_PARALLEL_BRANCHES='1', NS_AFTER_JOB='98765')
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [r for r in helpers.records(p) if r['action'] == 'submit']
            self.assertEqual(len(rows), 17)
            self.assertTrue(all(r['B3_AFTER_JOB'] == '98765' for r in rows))

    def test_backtest_has_no_repeated_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); env = helpers.fixture(p)
            r = run('--backtest', 'planejar', **env)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = helpers.records(p)
            self.assertEqual(len(rows), 17)
            self.assertEqual(len({r['PREDICTIONS_ROOT'] for r in rows}), 17)
            self.assertEqual(len({r['COMPOSITE_RUN_ID'] for r in rows}), 17)


if __name__ == '__main__':
    unittest.main()
