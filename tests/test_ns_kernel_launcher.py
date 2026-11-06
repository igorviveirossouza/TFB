"""Shell contract tests without Torch, Slurm, network or remote datasets."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/run_ns_kernels_isolado.sh"


class KernelLauncherTests(unittest.TestCase):
    def run_wrapper(self, *args, env=None, script=WRAPPER):
        merged = dict(os.environ)
        # Keep tests independent of an interactive experiment shell.
        for name in list(merged):
            if name.startswith("NS_") or name in {"MODEL", "TRAIN_VALUES", "LOOKBACKS_VALUES", "HORIZON_VALUES", "K_LOSS_VALUES", "POSITIVE_LAMBDAS", "CROSS_LOSSES_VALUES", "K_EVAL_VALUES", "DELTAS_VALUES", "KERNEL_SPECS", "EXPERIMENT_TAG", "TFB_ROOT"}:
                merged.pop(name)
        merged.update(env or {})
        return subprocess.run(["bash", str(script), *args], env=merged,
                              text=True, capture_output=True)

    def fixture(self, directory, real_launchers=False):
        root = Path(directory)
        (root / "scripts").mkdir()
        (root / "bin").mkdir()
        (root / "returns.csv").write_text("fixture\n")
        mock = '''import json, os, pathlib, sys
path = pathlib.Path(os.environ['CAPTURE'])
record = dict(action=sys.argv[1], **{k:os.environ.get(k) for k in ['RUN_TAG','MODELS','CROSS_LOSSES','CROSS_MASK','HORIZONS','TRADE_WINDOWS','LAMBDAS','B3_AFTER_JOB','GRES','MAX_CONCURRENT_JOBS']})
with path.open('a') as stream: stream.write(json.dumps(record)+'\\n')
if sys.argv[1] == 'submit':
    counter=path.with_suffix('.counter')
    job=int(counter.read_text())+1 if counter.exists() else 1001
    counter.write_text(str(job))
    print(f'Job {job}: array local 0-1; tarefas globais 0-1')
    if os.environ.get('FAIL_BRANCH') and os.environ['RUN_TAG'].endswith(os.environ['FAIL_BRANCH']): sys.exit(1)
'''
        (root / "mock.py").write_text(mock)
        for name in ("run_temporal_experiment.sh", "run_composite_trading_experiment.sh"):
            if real_launchers:
                shutil.copy(ROOT / "scripts" / name, root / "scripts" / name)
            else:
                (root / "scripts" / name).write_text('#!/usr/bin/env bash\nexec "$PYTHON_BIN" "$TFB_ROOT/mock.py" "$@"\n')
        shutil.copy(ROOT / "scripts/b3_experiment_env.sh", root / "scripts/b3_experiment_env.sh")
        (root / "scripts/b3_experiment.py").write_text("import os\nprint(os.environ['B3_INLINE_CONFIG'])\n")
        sbatch = root / "bin/sbatch"
        sbatch.write_text("#!/usr/bin/env bash\nexit 99\n")
        sbatch.chmod(0o755)
        script = root / "scripts/run_ns_kernels_isolado.sh"
        # Localize data roots only in the fixture, never in the product file.
        script.write_text(WRAPPER.read_text().replace('/snfs2/igor.viveiros', str(root / 'store')))
        env = dict(TFB_ROOT=str(root), NS_PYTHON_BIN=sys.executable,
                   NS_VENV_PATH=str(root / 'no_venv'), NS_RETURNS_FILE=str(root / 'returns.csv'),
                   CAPTURE=str(root / 'capture.jsonl'), PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'])
        return root, script, env

    def test_default_grade_and_alias_pool(self):
        r = self.run_wrapper('--plan')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('TOTAL: 160 treinos unicos; 14 ramos', r.stdout)
        self.assertIn('intermediate_band_2', r.stdout)
        self.assertIn('student 0.25 natural 10', r.stdout)
        self.assertIn('Ramos encadeados', r.stdout)

    def test_terminal_overrides_and_nondivisible_pairs(self):
        env=dict(MODEL='DUET TimesNet',HORIZON_VALUES='5 10',K_LOSS_VALUES='1 5',
                 KERNEL_SPECS='cauchy;student 10',NS_MASK_A='0.025',NS_GRES='',
                 CROSS_LOSSES_VALUES='hinge hinge_proximity')
        r=self.run_wrapper('--plan',env=env)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('TOTAL: 152 treinos',r.stdout)
        self.assertIn('cauchy 0.025 natural',r.stdout)
        bad=self.run_wrapper('--plan',env={'K_LOSS_VALUES':'5'})
        self.assertNotEqual(bad.returncode,0)
        self.assertIn('H=1 nao tem',bad.stderr)

    def test_no_duplicated_plain_loss_and_empty_gres_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,script,env=self.fixture(tmp)
            env['NS_GRES']=''
            r=self.run_wrapper('--dry-run',env=env,script=script)
            self.assertEqual(r.returncode,0,r.stderr)
            rows=[json.loads(s) for s in (root/'capture.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),14)
            self.assertEqual(sum(x['CROSS_LOSSES']=='ranknet' for x in rows),1)
            self.assertEqual(sum(x['CROSS_LOSSES']=='ranknet_proximity' for x in rows),12)
            self.assertTrue(all(x['GRES']=='' for x in rows))
            self.assertIn('compact 0.25 natural 2 3',[x['CROSS_MASK'].strip() for x in rows])

    def test_full_shape_reaches_inline_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,script,env=self.fixture(tmp,real_launchers=True)
            env['NS_ONLY_BRANCHES']='student_10 compact_2_3 cosine'
            r=self.run_wrapper('--dry-run',env=env,script=script)
            self.assertEqual(r.returncode,0,r.stderr)
            configs=[json.loads(s) for s in r.stdout.splitlines() if s.startswith('{')]
            self.assertEqual([x['loss']['loss_cross_mask'] for x in configs],
                [['student',0.25,'natural',10.0],['compact',0.25,'natural',2.0,3.0],['cosine',0.25,'natural']])
            self.assertTrue(all(x['horizons']==[1,25] for x in configs))

    def test_chained_submissions_and_repeated_train_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,script,env=self.fixture(tmp)
            env['NS_ONLY_BRANCHES']='temporal sem_kernel student_10'
            r=self.run_wrapper('--train',env=env,script=script)
            self.assertEqual(r.returncode,0,r.stderr)
            rows=[json.loads(s) for s in (root/'capture.jsonl').read_text().splitlines()]
            submitted=[x for x in rows if x['action']=='submit']
            self.assertEqual([x['B3_AFTER_JOB'] for x in submitted],['','1001','1002'])
            r2=self.run_wrapper('--train',env=env,script=script)
            self.assertEqual(r2.returncode,0,r2.stderr)
            rows=[json.loads(s) for s in (root/'capture.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['action']=='submit' for x in rows),3)
            changed=self.run_wrapper('--train',env=dict(env,NS_SEED='2027'),script=script)
            self.assertNotEqual(changed.returncode,0)
            self.assertIn('Parametros alterados',changed.stderr)

    def test_partial_submission_keeps_marker_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,script,env=self.fixture(tmp)
            env.update(NS_ONLY_BRANCHES='temporal sem_kernel student_10',FAIL_BRANCH='sem_kernel')
            r=self.run_wrapper('--train',env=env,script=script)
            self.assertNotEqual(r.returncode,0)
            r2=self.run_wrapper('--train',env=env,script=script)
            self.assertNotEqual(r2.returncode,0)
            self.assertIn('incerta/parcial',r2.stderr)
            rows=[json.loads(s) for s in (root/'capture.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['action']=='submit' for x in rows),2)

if __name__ == '__main__':
    unittest.main()
