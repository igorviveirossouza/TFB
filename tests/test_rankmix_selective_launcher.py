"""Selective proximity grid, frozen producer JSON and unchanged legacy controls."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selective_robust_helpers', ROOT/'tests/test_rankmix_robustez_launcher.py')
robust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robust)
WRAPPER = ROOT/'scripts/run_rankmix_proximity_seletiva.sh'
FIELD = 'loss_rank_mix_proximity_components'
SMALL = dict(LOOKBACKS_VALUES='32', HORIZON_VALUES='1', LAMBDAS_VALUES='0 .5 1')


def run(*args, **overrides):
    return subprocess.run(['bash', str(WRAPPER), *args], env=robust.environment(**overrides),
                          capture_output=True, text=True)


class SelectiveLauncherTest(unittest.TestCase):
    def check_total(self, total, **overrides):
        r = run('--plan', **overrides)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f'TOTAL: {total} treinos unicos', r.stdout)
        return r

    def test_default_paired_grid_and_editable_axes(self):
        r = self.check_total(78)
        self.assertIn('Controles lambda=0: 6; mistos: 54; ranking puro lambda=1: 18.', r.stdout)
        self.assertIn('cortes=90', r.stdout)
        self.assertIn('seeds=2028', r.stdout)
        self.assertIn('A=[0.01]', r.stdout)
        self.check_total(156, MODEL='iTransformer Nonstationary_Transformer')
        self.check_total(156, TRAIN_VALUES='80 90')
        self.check_total(54, RANK_MIX_PROXIMITY_SPECS='listnet')
        self.check_total(126, NS_MASK_A_VALUES='0.01 0.025')

    def test_selectors_are_validated_and_equivalent_active_scopes_deduplicated(self):
        self.check_total(78, RANK_MIX_PROXIMITY_SPECS='all;listmle listnet;listnet;listnet listnet')
        self.check_total(54, RANK_MIX_WEIGHT_SPECS='1 0 0')
        self.check_total(54, RANK_MIX_WEIGHT_SPECS='0 1 0')
        for scope in ('', 'all;', 'listnet bad', 'all listnet'):
            r = run('--plan', RANK_MIX_PROXIMITY_SPECS=scope)
            self.assertNotEqual(r.returncode, 0, scope)
            self.assertTrue('PROXIMITY_SPECS' in r.stderr or 'Seletor Proximity' in r.stderr, r.stderr)

    def test_defaults_file_and_export_overrides(self):
        reference = run('--print-exports')
        self.assertEqual(reference.returncode, 0, reference.stderr)
        command = '\n'.join(line for line in reference.stdout.splitlines() if line.startswith('export '))
        dump = '\npython3 -c \'import json,os; print(json.dumps(dict(os.environ)))\''
        expected = subprocess.run(['bash', '-c', command+dump], env=robust.environment(),
                                  text=True, capture_output=True, check=True)
        actual = subprocess.run(['bash', '-c', 'source "$1"'+dump, 'bash',
                                 str(ROOT/'scripts/config_rankmix_proximity_seletiva.defaults.sh')],
                                env=robust.environment(), text=True, capture_output=True, check=True)
        expected, actual = json.loads(expected.stdout), json.loads(actual.stdout)
        for line in command.splitlines():
            name = line.split('=', 1)[0].split()[1]
            self.assertEqual(actual[name], expected[name], name)
        changed = run('--print-exports', RANK_MIX_PROXIMITY_SPECS='listmle', NS_MASK_A_VALUES='0.025')
        self.assertIn('export RANK_MIX_PROXIMITY_SPECS=listmle', changed.stdout)
        self.assertIn('export NS_MASK_A_VALUES=0.025', changed.stdout)
        self.assertIn('bash scripts/run_rankmix_proximity_seletiva.sh --train', reference.stdout)
        self.assertNotIn('bash scripts/run_rankmix_robustez.sh --train', reference.stdout)

    def test_real_producer_freezes_scopes_and_isolates_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            env = robust.fixture(path)
            r = run('--dry-run', **env, **SMALL,
                    RANK_MIX_PROXIMITY_COMPONENTS='listmle', RANK_MIX_PROXIMITY_SELECTIVE='1')
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = robust.captured(path)
            self.assertEqual(len(rows), 7)
            self.assertEqual(len({row['root'] for row in rows}), 7)
            roots = [Path(row['root']) for row in rows]
            self.assertFalse(any(a in b.parents for a in roots for b in roots if a != b))
            scopes = [row['config']['loss'].get(FIELD) for row in rows]
            self.assertEqual(scopes.count(['listnet']), 2)
            self.assertEqual(scopes.count(None), 5)
            for row in rows:
                cfg = row['config']
                self.assertEqual(cfg['loss']['loss_cross_construction'], 'origin_anchored_windows')
                self.assertEqual(cfg['loss']['loss_rank_windows'], [1, 5, 10, 15, 20, 25])
                self.assertEqual(row['MOMENTUM_MODE'], 'off')
                self.assertEqual(row['MOMENTUM_INPUT_MODE'], 'off')
                self.assertEqual('prox_on=listnet' in row['root'], cfg['loss'].get(FIELD) == ['listnet'])
                self.assertEqual(row['B3_WORKER_LAUNCHER'], str(robust.WRAPPER))

    def test_default_producer_yields_78_distinct_fit_paths(self):
        spec = importlib.util.spec_from_file_location('selective_launcher_b3', ROOT/'scripts/b3_experiment.py')
        b3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            r = run('--dry-run', **robust.fixture(path))
            self.assertEqual(r.returncode, 0, r.stderr)
            paths, signatures, scopes = [], [], []
            for row in robust.captured(path):
                for task in b3.make_tasks(row['config'], row['argv'][1]):
                    _, hyper, _, _ = b3.model_spec(row['config'], task)
                    paths.append(str(Path(row['root'])/b3.output_path(task)))
                    signatures.append(json.dumps(task, sort_keys=True))
                    scopes.append(hyper.get(FIELD))
            self.assertEqual(len(paths), 78)
            self.assertEqual(len(set(paths)), 78)
            self.assertEqual(len(set(signatures)), 78)
            self.assertEqual(scopes.count(['listnet']), 24)
            self.assertEqual(sum('/proximity_on=listnet/' in p for p in paths), 24)

    def test_multiple_a_only_duplicates_proximity(self):
        settings = {**SMALL, 'NS_MASK_A_VALUES': '.01 .025', 'TEMPORAL_LOSSES_VALUES': 'mse huber'}
        self.check_total(17, **settings)  # two controls + 2*5 mixed + five pure rankings
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            r = run('--dry-run', **robust.fixture(path), **settings)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = robust.captured(path)
            controls = [row for row in rows if row['argv'][1] == 'temporal']
            self.assertEqual(len(controls), 2)
            self.assertEqual(len(rows), 17)
            self.assertEqual(len({row['root'] for row in rows}), 17)

    def test_submission_resume_selects_scope_and_frozen_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            env = robust.fixture(path)
            r = run('--train', **env, **SMALL, NS_AFTER_JOB='12345')
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = robust.captured(path, 'submit')
            self.assertEqual(len(rows), 7)
            self.assertEqual([row['B3_AFTER_JOB'] for row in rows], ['12345']+[str(x) for x in range(1001, 1007)])
            again = run('--train', **env, **SMALL, NS_AFTER_JOB='12345')
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(len(robust.captured(path, 'submit')), 7)
            listing = run('--list-branches', **SMALL)
            branch = next(line.split(' | ')[0] for line in listing.stdout.splitlines() if 'prox_on_listnet' in line)
            self.check_total(1, **SMALL, NS_ONLY_BRANCHES=branch)
            selective = next(row for row in rows if row['config']['loss'].get(FIELD))
            worker = run('--on-node', 'composto', 'worker', '--offset', '0', **env,
                         PYTHON_BIN=env['NS_PYTHON_BIN'], B3_INLINE_CONFIG=json.dumps(selective['config']),
                         B3_CONFIG_SOURCE=selective['B3_CONFIG_SOURCE'],
                         RANK_MIX_PROXIMITY_COMPONENTS='listmle', RANK_MIX_PROXIMITY_SELECTIVE='1')
            self.assertEqual(worker.returncode, 0, worker.stderr)
            self.assertEqual(robust.captured(path)[-1]['config'], selective['config'])


if __name__ == '__main__':
    unittest.main()
