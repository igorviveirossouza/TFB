"""Focused hypotheses, matched references, producer contracts and resumability."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rankmix_refinement_helpers', ROOT/'tests/test_rankmix_refinement_launcher.py')
refinement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refinement)
helpers = refinement.helpers
WRAPPER = ROOT/'scripts/run_ns_rankmix_refinamento_fino.sh'


def run(*args, **overrides):
    return subprocess.run(['bash', str(WRAPPER), *args],
                          env=helpers.environment(**overrides), capture_output=True, text=True)


def producer_records(path, **settings):
    env = helpers.fixture(path)
    for name in ['run_temporal_experiment.sh', 'run_composite_trading_experiment.sh', 'b3_experiment_env.sh']:
        shutil.copy(ROOT/'scripts'/name, path/'scripts'/name)
    (path/'scripts/b3_experiment.py').write_text(
        "import json,os,sys\nprint(json.dumps(dict(config=json.loads(os.environ['B3_INLINE_CONFIG']), root=os.environ['OUT_ROOT'], argv=sys.argv)))\n")
    result = run('--dry-run', **env, **settings)
    if result.returncode:
        raise AssertionError(result.stderr)
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]


class FineLauncherTest(unittest.TestCase):
    def check_total(self, n, **env):
        result = run('--plan', **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'TOTAL: {n} treinos unicos', result.stdout)
        return result

    def test_default_and_editable_axes(self):
        r = self.check_total(102)
        self.assertIn('Controles lambda=0: 6; mistos: 54; ranking puro lambda=1: 42', r.stdout)
        self.assertIn('particao=gorgonas_dev; GRES=vazio; limite=01:30:00', r.stdout)
        self.check_total(72, RANK_MIX_INCLUDE_REFERENCES='0')
        self.check_total(204, MODEL='Nonstationary_Transformer DUET')
        self.check_total(204, SEEDS_VALUES='2026 2027')
        self.check_total(162, TEMPORAL_LOSSES_VALUES='mse huber')
        self.check_total(34, LOOKBACKS_VALUES='256')
        self.check_total(198, K_LOSS_VALUES='1 5 25')
        self.check_total(156, KERNEL_SPECS='cauchy;gaussian')
        self.check_total(17, LOOKBACKS_VALUES='32', HORIZON_VALUES='5', K_LOSS_VALUES='5')

    def test_tuples_are_authoritative_over_inherited_old_grid(self):
        self.check_total(102, RANK_MIX_STAGE='composicao regional',
                         RANK_MIX_MODES='global regional', LAMBDAS_VALUES='0 .5 .8 1',
                         RANK_MIX_WEIGHT_SPECS='.5 .5 0', NS_MASK_A='0.25',
                         NS_MASK_A_VALUES='.025 .1')
        self.check_total(42, RANK_MIX_INCLUDE_REFERENCES='0',
                         RANK_MIX_FOCUS_SPECS='chosen|global|0 1 0|.2 .3|.01')
        self.check_total(60, RANK_MIX_INCLUDE_REFERENCES='0',
                         RANK_MIX_FOCUS_SPECS='chosen|global|0 1 0|.2 .3|.01 .02')

    def test_invalid_design_fails_before_submission(self):
        invalid = ['', 'x|global|.5 .5 0|.1', 'x|global|.5 .5 0|.1|.01|extra',
                   'x|unknown|.5 .5 0|.1|.01', 'x|global|.1 .1 0|.1|.01',
                   'x|global|.5 .5 0|0|.01', 'x|global|.5 .5 0|1|.01',
                   'x|global|.5 .5 0|.1 .10|.01', 'x|global|.5 .5 0|.1|0',
                   'x|global|.5 .5 0|.1|.01 .010', 'x|global|.5 .5 0|.1|nan',
                   'x|global|.5 .5 0|.1|.01;x|regional|.5 .5 0|.1|.01']
        for value in invalid:
            with self.subTest(value=value):
                self.assertNotEqual(run('--plan', RANK_MIX_FOCUS_SPECS=value).returncode, 0)
        for env in [dict(MOMENTUM_MODE='residual_output'), dict(MOMENTUM_INPUT_MODE='feature'),
                    dict(RANK_MIX_INCLUDE_REFERENCES='yes'), dict(NS_ONLY_BRANCHES='missing'),
                    dict(HORIZON_VALUES='1 25', K_LOSS_VALUES='5')]:
            with self.subTest(env=env):
                self.assertNotEqual(run('--plan', **env).returncode, 0)

    def test_real_producer_has102_unique_fits_and_exact_scientific_tuples(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = producer_records(Path(tmp))
            tasks = refinement.task_rows(rows)
            self.assertEqual(len(tasks), 102)
            self.assertEqual(len({path for path, _ in tasks}), 102)
            self.assertEqual(len({key for _, key in tasks}), 102)
            self.assertEqual(len(rows), 17)
            w_global, w_reg, w_ref = (.25,.75,0), (0,.9,.1), (0,.75,.25)
            expected = {('global', w_global, .05, .01),
                        ('global', w_global, .1, .005),
                        ('global', w_global, .1, .01),
                        ('global', w_global, 1., .01),
                        ('global', w_global, 1., .005)}
            expected |= {('regional', w, lam, .01) for w in [w_reg, w_ref] for lam in [.5, 1.]}
            actual = {(key[12], key[11], key[6], key[10]) for _, key in tasks
                      if key[5] == 'rankmix_proximity'}
            self.assertEqual(actual, expected)
            self.assertEqual(sum(key[6] == 0 for _, key in tasks), 6)
            plain = {(key[12], key[11], key[6]) for _, key in tasks if key[5] == 'rankmix'}
            self.assertEqual(plain, {(mode, weights, lam) for mode, weights, lam, _ in expected})
            for row in rows:
                self.assertEqual(row['config']['loss'].get('momentum_mode', 'off'), 'off')
                self.assertEqual(row['config']['loss'].get('momentum_input_mode', 'off'), 'off')

    def test_reference_toggle_reuses_submitted_controls_without_old_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = helpers.fixture(path)
            mock = path/'mock.py'
            mock.write_text(mock.read_text().replace("'PARTITION']", "'PARTITION','TIME_LIMIT']"))
            first = run('--train', **env, RANK_MIX_INCLUDE_REFERENCES='0')
            self.assertEqual(first.returncode, 0, first.stderr)
            first_rows = [r for r in helpers.records(path) if r['action'] == 'submit']
            self.assertEqual(len(first_rows), 12)
            self.assertTrue(all(r['TIME_LIMIT'] == '01:30:00' and r['GRES'] == '' and
                                r['MAX_CONCURRENT_JOBS'] == '3' for r in first_rows))
            self.assertTrue(all(r['MOMENTUM_MODE'] == r['MOMENTUM_INPUT_MODE'] == 'off'
                                for r in first_rows))
            second = run('--train', **env)
            self.assertEqual(second.returncode, 0, second.stderr)
            added = [r for r in helpers.records(path) if r['action'] == 'submit'][12:]
            self.assertEqual(len(added), 5)
            self.assertEqual([r['B3_AFTER_JOB'] for r in added], ['', '1013', '1014', '1015', '1016'])
            retry = run('--train', **env)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(len([r for r in helpers.records(path) if r['action'] == 'submit']), 17)
            changed = run('--train', **env, NS_BATCH_SIZE='8')
            self.assertNotEqual(changed.returncode, 0)

    def test_selection_backtest_and_parallel_routing(self):
        result = self.check_total(102)
        branch = next(line.split()[0] for line in result.stdout.splitlines()
                      if '_lambda0p05' in line and '_sem_kernel_' in line)
        self.check_total(6, NS_ONLY_BRANCHES=branch)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = helpers.fixture(path)
            r = run('--train', **env, NS_PARALLEL_BRANCHES='1', NS_AFTER_JOB='99999',
                    NS_PARTITION='medusas_dev', NS_GRES='gpu:1')
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [r for r in helpers.records(path) if r['action'] == 'submit']
            self.assertEqual(len(rows), 17)
            self.assertTrue(all(r['B3_AFTER_JOB'] == '99999' and r['GRES'] == 'gpu:1' for r in rows))
            r = run('--backtest', 'planejar', **env)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = [r for r in helpers.records(path) if r['action'] == 'planejar']
            self.assertEqual(len(rows), 17)
            self.assertEqual(len({r['PREDICTIONS_ROOT'] for r in rows}), 17)
            self.assertEqual(len({r['COMPOSITE_RUN_ID'] for r in rows}), 17)


if __name__ == '__main__':
    unittest.main()
