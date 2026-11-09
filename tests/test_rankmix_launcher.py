"""Editable rank-mixture grid and submission routing; no Slurm jobs are run."""
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'scripts/run_ns_rankmix_isolado.sh'


def environment(**overrides):
    env = dict(os.environ)
    names = {'MODEL', 'TRAIN_VALUES', 'LOOKBACKS_VALUES', 'HORIZON_VALUES',
             'K_LOSS_VALUES', 'POSITIVE_LAMBDAS', 'LAMBDAS_VALUES',
             'TEMPORAL_LOSSES_VALUES', 'SEEDS_VALUES', 'CROSS_LOSSES_VALUES',
             'K_EVAL_VALUES', 'DELTAS_VALUES', 'KERNEL_SPECS', 'EXPERIMENT_TAG',
             'TFB_ROOT', 'PRED_ROOT', 'BACKTEST_ROOT'}
    for name in list(env):
        if name.startswith(('NS_', 'MOMENTUM_', 'RANK_MIX_')) or name in names:
            env.pop(name)
    env.update(overrides)
    return env


def run(*args, **env):
    return subprocess.run(['bash', str(WRAPPER), *args], env=environment(**env),
                          text=True, capture_output=True)


def fixture(path):
    (path / 'scripts').mkdir()
    (path / 'bin').mkdir()
    (path / 'market').mkdir()
    (path / 'returns.csv').write_text('fixture\n')
    (path / 'strategies.json').write_text('{}\n')
    (path / 'bin/sbatch').write_text('#!/bin/bash\nexit 99\n')
    (path / 'bin/sbatch').chmod(0o755)
    (path / 'mock.py').write_text('''import json, os, pathlib, sys
path = pathlib.Path(os.environ['CAPTURE'])
keys = ['RUN_TAG','OUT_ROOT','TEMPORAL_LOSS','SEED','MODELS','LOOKBACKS','CROSS_LOSSES',
        'CROSS_MASK','HORIZONS','TRADE_WINDOWS','LAMBDAS','B3_AFTER_JOB','GRES',
        'MAX_CONCURRENT_JOBS','MOMENTUM_MODE','MOMENTUM_INPUT_MODE','RANK_MIX_WEIGHTS',
        'RANK_MIX_MODE','RANK_MIX_REGIONS','RANK_MIX_FRACTION','PREDICTIONS_ROOT',
        'RUN_ID','COMPOSITE_RUN_ID','K_EVAL_VALUES','PARTITION']
row = dict(action=sys.argv[1], **{k:os.environ.get(k) for k in keys})
with path.open('a') as stream: stream.write(json.dumps(row)+'\\n')
if sys.argv[1] == 'submit':
    counter = path.with_suffix('.counter')
    job = int(counter.read_text())+1 if counter.exists() else 1001
    counter.write_text(str(job))
    print(f'Job {job}: array local 0-1; tarefas globais 0-1')
    if os.environ.get('FAIL_BRANCH') and os.environ['RUN_TAG'].endswith(os.environ['FAIL_BRANCH']):
        sys.exit(1)
''')
    for name in ('run_temporal_experiment.sh', 'run_composite_trading_experiment.sh',
                 'run_backtests_cpu.sh', 'run_backtests_composto_cpu.sh'):
        (path / 'scripts' / name).write_text('#!/bin/bash\nexec "$PYTHON_BIN" "$TFB_ROOT/mock.py" "$@"\n')
    return dict(TFB_ROOT=str(path), NS_PYTHON_BIN=sys.executable,
                NS_VENV_PATH=str(path/'no_venv'), NS_RETURNS_FILE=str(path/'returns.csv'),
                NS_MARKET_DIR=str(path/'market'), NS_STRATEGIES_FILE=str(path/'strategies.json'),
                CAPTURE=str(path/'capture.jsonl'), PRED_ROOT=str(path/'pred'),
                BACKTEST_ROOT=str(path/'backtest'),
                PATH=str(path/'bin')+os.pathsep+os.environ['PATH'])


def records(path):
    return [json.loads(line) for line in (path/'capture.jsonl').read_text().splitlines()]


class RankMixLauncherTest(unittest.TestCase):
    def check_total(self, total, **env):
        result = run('--plan', **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'TOTAL: {total} treinos unicos', result.stdout)
        return result

    def test_default_counts_share_controls_and_pure_endpoints(self):
        r = self.check_total(132)
        self.assertIn('Controles lambda=0: 12; mistos: 96; ranking puro lambda=1: 24', r.stdout)
        self.assertIn('14 ramos selecionados', r.stdout)
        self.assertIn('Momentum: benchmark externo, 0 treinos', r.stdout)
        self.assertIn('[0.5 0.5 0]', r.stdout)

    def test_modes_and_lambdas_can_be_selected(self):
        for mode in ['global', 'regional']:
            with self.subTest(mode=mode): self.check_total(72, RANK_MIX_MODES=mode)
        for lambdas, n in [('0',12), ('.5 .8',96), ('1',24), ('0 1',36)]:
            with self.subTest(lambdas=lambdas): self.check_total(n, LAMBDAS_VALUES=lambdas)

    def test_model_seed_and_temporal_counts(self):
        self.check_total(264, MODEL='Nonstationary_Transformer simpleFORMER')
        self.check_total(264, SEEDS_VALUES='2026 2027')
        self.check_total(78, TEMPORAL_LOSSES_VALUES='mse')

    def test_valid_h_k_pairs_and_kernel_reuse(self):
        self.check_total(252, K_LOSS_VALUES='1 5 25')
        self.check_total(192, KERNEL_SPECS='cauchy;gaussian')
        self.check_total(132, DELTAS_VALUES='0 .01 .1')
        self.assertNotEqual(run('--plan', HORIZON_VALUES='1 25', K_LOSS_VALUES='25').returncode, 0)

    def test_third_component_and_unequal_weights(self):
        self.check_total(132, RANK_MIX_WEIGHTS='0 .5 .5')
        self.check_total(132, RANK_MIX_WEIGHTS='.2 .3 .5', RANK_MIX_REGIONS='bottom top top')

    def test_invalid_parameters_fail_before_data_or_submission(self):
        invalid = [dict(RANK_MIX_WEIGHTS=x) for x in ['.5 .5', '0 0 0', '.4 .4 0', '-.1 .6 .5', 'nan .5 .5', 'inf 0 0', '1.1 0 0']]
        invalid += [dict(RANK_MIX_MODES=x) for x in ['unknown', 'global global']]
        invalid += [dict(RANK_MIX_REGIONS=x) for x in ['top bottom', 'top middle bottom']]
        invalid += [dict(RANK_MIX_FRACTION=x) for x in ['0', '.51', 'nan', '-.1']]
        invalid += [dict(CROSS_LOSSES_VALUES=x) for x in ['rankmix', 'rankmix_proximity', 'listnet listnet_proximity']]
        invalid += [dict(MOMENTUM_MODE='residual_output'), dict(MOMENTUM_INPUT_MODE='feature'), dict(LAMBDAS_VALUES='0 .5 .50 1')]
        for env in invalid:
            with self.subTest(env=env): self.assertNotEqual(run('--plan', **env).returncode, 0)

    def test_dry_run_routes_metadata_and_deduplicates_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); env = fixture(p)
            r = run('--dry-run', **env, NS_GRES='', RANK_MIX_WEIGHTS='0 .5 .5')
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = records(p)
            self.assertEqual(len(rows), 14)
            self.assertEqual(len({r['OUT_ROOT'] for r in rows}), 14)
            controls = [r for r in rows if not r['CROSS_LOSSES']]
            self.assertEqual(len(controls), 2)
            self.assertTrue(all(r['RANK_MIX_MODE'] is None and r['RANK_MIX_WEIGHTS'] is None for r in controls))
            composed = [r for r in rows if r['CROSS_LOSSES']]
            self.assertEqual({r['RANK_MIX_MODE'] for r in composed}, {'global', 'regional'})
            self.assertTrue(all(r['RANK_MIX_WEIGHTS'] == '0 0.5 0.5' for r in composed))
            self.assertTrue(all('/mistura='+r['RANK_MIX_MODE']+'_' in r['OUT_ROOT'] for r in composed))
            self.assertTrue(all(r['GRES'] == '' and r['MOMENTUM_MODE'] == r['MOMENTUM_INPUT_MODE'] == 'off' for r in rows))
            effective = []
            for row in rows:
                for lb in map(int, row['LOOKBACKS'].split()):
                    for h in map(int, row['HORIZONS'].split()):
                        ks = [k for k in map(int,row['TRADE_WINDOWS'].split()) if k<=h and h%k==0] if row['CROSS_LOSSES'] else [None]
                        for k in ks:
                            for lam in map(float, row['LAMBDAS'].split() or ['0']):
                                effective.append((lb,h,k,lam,row['TEMPORAL_LOSS'] if lam<1 else None,row['CROSS_LOSSES'],row['RANK_MIX_MODE']))
            self.assertEqual(len(effective),132)
            self.assertEqual(len(set(effective)),132)

    def test_signature_changes_with_parameters_but_not_number_spelling(self):
        def signature(**env):
            r = run('--plan', **env)
            self.assertEqual(r.returncode,0,r.stderr)
            return re.search(r'assinatura: ([a-f0-9]+)',r.stdout).group(1)
        base=signature()
        self.assertEqual(base,signature(RANK_MIX_WEIGHTS='.50 5e-1 0.0'))
        self.assertNotEqual(base,signature(RANK_MIX_WEIGHTS='0 .5 .5'))
        self.assertNotEqual(base,signature(RANK_MIX_REGIONS='top bottom bottom'))
        self.assertNotEqual(base,signature(RANK_MIX_FRACTION='.25'))

    def test_real_launchers_produce_same132_unique_scientific_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); env=fixture(p)
            for name in ['run_temporal_experiment.sh', 'run_composite_trading_experiment.sh', 'b3_experiment_env.sh']:
                shutil.copy(ROOT/'scripts'/name, p/'scripts'/name)
            (p/'scripts/b3_experiment.py').write_text("import json, os, sys\nprint(json.dumps(dict(config=json.loads(os.environ['B3_INLINE_CONFIG']), root=os.environ['OUT_ROOT'], argv=sys.argv)))\n")
            r=run('--dry-run',**env,RANK_MIX_WEIGHTS='.2 .3 .5',NS_GRES='')
            self.assertEqual(r.returncode,0,r.stderr)
            rows=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')]
            self.assertEqual(len(rows),14)
            spec=importlib.util.spec_from_file_location('rankmix_launcher_b3',ROOT/'scripts/b3_experiment.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            paths=[]; effective=[]
            for row in rows:
                config=row['config']; mode=row['argv'][1]
                for task in module.make_tasks(config,mode):
                    paths.append(str(Path(row['root'])/module.output_path(task)))
                    _,hyper,_,_=module.model_spec(config,task)
                    key=(task['modelo'],task['lookback'],task['pred_len'],task['k'],
                         task['loss'],task['cross_lambda'],config['loss']['loss_temporal'] if task['cross_lambda']<1 else None,
                         config['strategy']['seed'],task.get('mask_type'),hyper.get('loss_rank_mix_mode'))
                    effective.append(key)
                    if mode=='composto':
                        self.assertEqual(hyper['loss_rank_mix_weights'],[.2,.3,.5])
                        self.assertEqual(hyper['loss_rank_mix_regions'],['top','bottom','top'])
                        self.assertIn(hyper['loss_rank_mix_mode'],['global','regional'])
                    else:
                        self.assertFalse(any(k.startswith('loss_rank_mix_') for k in hyper))
            self.assertEqual(len(paths),132)
            self.assertEqual(len(set(paths)),132)
            self.assertEqual(len(set(effective)),132)

    def test_submission_resume_and_parallel_dependency(self):
        for parallel in ['0','1']:
            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp); env=fixture(p)
                r=run('--train',**env,NS_PARALLEL_BRANCHES=parallel,NS_MAX_CONCURRENT_JOBS='3')
                self.assertEqual(r.returncode,0,r.stderr)
                rows=[r for r in records(p) if r['action']=='submit']
                self.assertEqual(len(rows),14)
                deps=['']*14 if parallel=='1' else ['']+[str(n) for n in range(1001,1014)]
                self.assertEqual([r['B3_AFTER_JOB'] for r in rows],deps)
                self.assertTrue(all(r['MAX_CONCURRENT_JOBS']=='3' for r in rows))
                retry=run('--train',**env,NS_PARALLEL_BRANCHES=parallel,NS_MAX_CONCURRENT_JOBS='3')
                self.assertEqual(retry.returncode,0,retry.stderr)
                self.assertEqual(len([r for r in records(p) if r['action']=='submit']),14)

    def test_partial_submission_remains_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); env=fixture(p)
            r=run('--train',**env,FAIL_BRANCH='composto/sem_kernel')
            self.assertNotEqual(r.returncode,0)
            retry=run('--train',**env)
            self.assertNotEqual(retry.returncode,0)
            self.assertIn('incerta/parcial',retry.stderr)
            self.assertEqual(len([r for r in records(p) if r['action']=='submit']),2)

    def test_backtest_reads_each_prediction_branch_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); env=fixture(p)
            r=run('--backtest','planejar',**env,K_EVAL_VALUES='1 10 25')
            self.assertEqual(r.returncode,0,r.stderr)
            rows=records(p)
            self.assertEqual(len(rows),14)
            self.assertEqual(len({r['PREDICTIONS_ROOT'] for r in rows}),14)
            self.assertEqual(len({r['COMPOSITE_RUN_ID'] for r in rows}),14)
            self.assertTrue(all(r['K_EVAL_VALUES']=='1 10 25' for r in rows))


if __name__ == '__main__':
    unittest.main()
