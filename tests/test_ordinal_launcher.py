"""Ordinal design, effective settings, frozen workers and submission safeguards."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import b3_experiment as b3
from scripts import ordinal_experiment as ordinal


def settings(**values):
    return ordinal.settings({k: str(v) for k, v in values.items()})


def tasks(branches):
    return [(b, t) for b in branches for t in b['tasks']]


class OrdinalLauncherTest(unittest.TestCase):
    def test_default_grid_has_one_temporal_control_per_architecture_context(self):
        branches = ordinal.build_branches(settings())
        rows = tasks(branches)
        self.assertEqual(len(rows), 102)
        self.assertEqual(sum(t['cross_lambda'] == 0 for _, t in rows), 6)
        self.assertEqual(sum(t['cross_lambda'] == 1 for _, t in rows), 24)
        self.assertEqual({t['modelo'] for _, t in rows}, {'Nonstationary_Transformer'})
        self.assertEqual({t['lookback'] for _, t in rows}, {32, 256, 512})
        self.assertEqual({t['pred_len'] for _, t in rows}, {1, 25})
        for b, t in rows:
            if t['cross_lambda']:
                self.assertEqual(t['rank_windows'], [k for k in [1,5,10,15,20,25] if k <= t['pred_len']])
                self.assertEqual(t['bucket_count'], 3)
            else:
                self.assertNotIn('loss_bucket_count', b['config']['loss'])
        paths = [str(Path(b['relative'])/b3.output_path(t)) for b, t in rows]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertEqual(sum(p.startswith('loss=') for p in Path(path).parts), 1)

    def test_multiple_buckets_models_seeds_do_not_repeat_temporal_controls(self):
        branches = ordinal.build_branches(settings(BUCKET_COUNTS_VALUES='3 10 3',
            MODEL='NonStationary_Transformer simpleFORMER', SEEDS_VALUES='2026 2027',
            LAMBDAS_VALUES='0 0 .5 .50 1'))
        rows = tasks(branches)
        self.assertEqual(sum(t['cross_lambda'] == 0 for _, t in rows), 24)
        self.assertEqual(len(rows), 408)
        controls = [(b['config']['strategy']['seed'], t['modelo'], t['treino'], t['lookback'], t['pred_len'])
                    for b,t in rows if t['cross_lambda']==0]
        self.assertEqual(len(controls), len(set(controls)))

    def test_blocks_respects_visibility_and_does_not_require_origin_windows(self):
        branches = ordinal.build_branches(settings(NS_RANKING_PROTOCOL='blocks',
            HORIZON_VALUES='1 5 25', K_LOSS_VALUES='1 5 25', RANK_WINDOWS_VALUES='unused',
            LOOKBACKS_VALUES='32', LAMBDAS_VALUES='0 1'))
        rows = tasks(branches)
        self.assertEqual(len(rows), 27)
        self.assertEqual({(t['pred_len'], t['k']) for _,t in rows if t['cross_lambda']},
                         {(1,1),(5,1),(5,5),(25,1),(25,5),(25,25)})
        self.assertTrue(all(not t.get('rank_windows') for _,t in rows))

    def test_hyperparameters_reach_native_model_and_active_paths(self):
        branches = ordinal.build_branches(settings(LOOKBACKS_VALUES='32', HORIZON_VALUES='1',
            LAMBDAS_VALUES='.7', NS_ORDINAL_TAU='.4', NS_BUCKET_TAU='.6',
            NS_BUCKET_TARGET_TAU='.8', NS_HINGE_MARGIN='0', NS_BATCH_SIZE='8',
            NS_NUM_EPOCHS='11', NS_PATIENCE='4', NS_LEARNING_RATE='.002', NS_LRADJ='constant'))
        for b,t in tasks(branches):
            factory, hyper, adapter, _ = b3.model_spec(b['config'],t)
            self.assertEqual(factory, 'ordinal_model_adapter.OrdinalModel')
            self.assertIsNone(adapter)
            self.assertEqual(hyper['_ordinal_model_adapter'], 'transformer_adapter')
            self.assertEqual(hyper['batch_size'],8)
            self.assertEqual(hyper['num_epochs'],11)
            self.assertEqual(hyper['patience'],4)
            self.assertEqual(hyper['lr'],.002)
            self.assertEqual(hyper['lradj'],'constant')
            self.assertEqual(hyper['loss_bucket_count'],3)
            path = str(b3.output_path(t))
            if t['loss']=='ordinal_ce':
                self.assertEqual(hyper['loss_ordinal_tau'],.4)
                self.assertIn('ordinal_tau=0.4',path)
                self.assertNotIn('loss_bucket_tau',hyper)
            elif t['loss']=='bucket_hinge':
                self.assertEqual(hyper['loss_hinge_margin'],0)
                self.assertIn('bucket_margin=0.0',path)
            elif t['loss']=='bucket_listnet':
                self.assertEqual(hyper['loss_bucket_tau'],.6)
                self.assertEqual(hyper['loss_bucket_target_tau'],.8)

    def test_model_override_merges_without_hiding_other_architectures(self):
        branches = ordinal.build_branches(settings(MODEL='DUET PatchTST DTAF',
            NS_MODEL_DEFAULTS_JSON='{"Nonstationary_Transformer":{"d_model":64},"PatchTST":{"d_model":32}}'))
        cfg = branches[0]['config']
        self.assertEqual(set(cfg['models']), {'DUET', 'PatchTST', 'DTAF'})
        self.assertEqual(cfg['model_defaults']['DUET']['d_model'],32)
        self.assertEqual(cfg['model_defaults']['DUET']['num_experts'],4)
        self.assertEqual(cfg['model_defaults']['PatchTST']['d_model'],32)
        self.assertTrue(cfg['model_defaults']['DTAF']['norm'])
        self.assertNotIn('d_model',cfg['model_defaults']['DTAF'])
        self.assertEqual(cfg['model_specs']['PatchTST']['adapter'],'transformer_adapter')
        self.assertIsNone(cfg['model_specs']['DTAF']['adapter'])

    def test_qualified_aliases_deduplicate_and_overrides_merge_with_old_defaults(self):
        branches = ordinal.build_branches(settings(MODEL='NonStationary_Transformer time_series_library.Nonstationary_Transformer ts_benchmark.baselines.time_series_library.models.Nonstationary_Transformer.Nonstationary_Transformer',
            NS_MODEL_DEFAULTS_JSON='{"time_series_library.Nonstationary_Transformer":{"d_model":64}}'))
        self.assertEqual(len(tasks(branches)),102)
        cfg = branches[0]['config']
        self.assertEqual(cfg['models'],['Nonstationary_Transformer'])
        self.assertEqual(cfg['model_defaults']['Nonstationary_Transformer']['d_model'],64)
        self.assertEqual(cfg['model_defaults']['Nonstationary_Transformer']['p_hidden_dims'],[32,32])

    def test_full_source_class_and_exported_statistical_models_are_discoverable(self):
        branches = ordinal.build_branches(settings(MODEL='time_series_library.models.TiDE.TiDE darts.ARIMA',
            NS_MODEL_DEFAULTS_JSON=''))
        cfg = branches[0]['config']
        self.assertIn('time_series_library.models.TiDE.TiDE',cfg['models'])
        self.assertIn('ARIMA',cfg['models'])
        self.assertEqual(cfg['model_specs']['ARIMA']['family'],'darts')
        self.assertEqual(cfg['model_specs']['time_series_library.models.TiDE.TiDE']['adapter'],'transformer_adapter')
        # Discovery says nothing about whether a native training loop can use
        # ordinal labels: the producer's runtime preflight makes that decision.

    def test_pure_ordinal_not_repeated_for_inactive_temporal_loss(self):
        branches = ordinal.build_branches(settings(TEMPORAL_LOSSES_VALUES='mse huber',
            LOOKBACKS_VALUES='32',HORIZON_VALUES='1',LAMBDAS_VALUES='0 .5 1'))
        rows=tasks(branches)
        self.assertEqual(sum(t['cross_lambda']==0 for _,t in rows),2)
        self.assertEqual(sum(t['cross_lambda']==.5 for _,t in rows),8)
        self.assertEqual(sum(t['cross_lambda']==1 for _,t in rows),4)

    def test_invalid_settings_fail_before_any_submission(self):
        for opts in [dict(BUCKET_COUNTS_VALUES='1'),dict(NS_ORDINAL_TAU='0'),
            dict(NS_BUCKET_TAU='nan'),dict(NS_HINGE_MARGIN='-.1'),dict(LAMBDAS_VALUES='1.1'),
            dict(NS_RANKING_PROTOCOL='bad'),dict(K_LOSS_VALUES='5'),dict(MODEL='missing'),
            dict(NS_MODEL_DEFAULTS_JSON='{"Nonstationary_Transformer":{"loss_cross":"mse"}}')]:
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                ordinal.build_branches(settings(**opts))

    def test_actual_dry_run_produces_slurm_resources_and_frozen_configs(self):
        env=dict(os.environ)
        env.update(settings(LOOKBACKS_VALUES='32',HORIZON_VALUES='1',LAMBDAS_VALUES='0 1',
            NS_PARTITION='medusas_shr',NS_GRES='gpu:1',NS_CPUS_PER_TASK='5',NS_MEMORY='20G',
            NS_MAX_CONCURRENT_JOBS='3'))
        result=subprocess.run(['bash',str(ROOT/'scripts/run_ordinal_experiment.sh'),'--dry-run'],
            env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('TOTAL: 5 treinos unicos',result.stdout)
        self.assertIn('-p medusas_shr',result.stdout)
        self.assertIn('--gres=gpu:1',result.stdout)
        self.assertIn('--mem=20G',result.stdout)
        self.assertIn('--cpus-per-task=5',result.stdout)
        self.assertIn('--on-node composto worker',result.stdout)
        b=ordinal.build_branches(settings())[1]
        with patch.dict(os.environ,{'B3_INLINE_CONFIG':'stale','B3_AFTER_JOB':'999'}):
            child=ordinal.branch_environment(settings(),b,'123')
        self.assertEqual(json.loads(child['B3_INLINE_CONFIG']),b['config'])
        self.assertEqual(child['B3_AFTER_JOB'],'123')

    def test_submission_is_serial_and_retry_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=settings(NS_LOG_ROOT=tmp,NS_PYTHON_BIN=sys.executable,
                LOOKBACKS_VALUES='32',HORIZON_VALUES='1',LAMBDAS_VALUES='0 1')
            branches=ordinal.build_branches(s); seen=[]
            def producer(s,b,action,dependency=''):
                seen.append((action,dependency)); return [str(1000+len(seen))] if action=='submit' else []
            with patch.object(ordinal.shutil,'which',return_value='/bin/true'),patch.object(ordinal,'run_producer',side_effect=producer):
                with contextlib.redirect_stdout(io.StringIO()): ordinal.train(s,branches)
                submits=[x for x in seen if x[0]=='submit']
                self.assertEqual(submits[0],('submit',''))
                self.assertTrue(all(x[1] for x in submits[1:]))
                seen.clear()
                with contextlib.redirect_stdout(io.StringIO()): ordinal.train(s,branches)
                self.assertFalse(any(x[0]=='submit' for x in seen))

    def test_interrupted_resume_requires_explicit_saved_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=settings(NS_LOG_ROOT=tmp,NS_PYTHON_BIN=sys.executable,
                LOOKBACKS_VALUES='32',HORIZON_VALUES='1',LAMBDAS_VALUES='0 1')
            branches=ordinal.build_branches(s)
            with patch.object(ordinal.shutil,'which',return_value='/bin/true'),patch.object(ordinal,'run_producer',return_value=['123']):
                with contextlib.redirect_stdout(io.StringIO()): ordinal.train(s,branches[:1])
                with self.assertRaisesRegex(ValueError,'NS_RESUME_CHAIN_OLD_JOBS'):
                    ordinal.train(s,branches)
            seen=[]
            def producer(s,b,action,dependency=''):
                seen.append((action,dependency));return ['124'] if action=='submit' else []
            s['NS_RESUME_CHAIN_OLD_JOBS']='1'
            with patch.object(ordinal.shutil,'which',return_value='/bin/true'),patch.object(ordinal,'run_producer',side_effect=producer):
                with contextlib.redirect_stdout(io.StringIO()): ordinal.train(s,branches)
            self.assertEqual(next(d for a,d in seen if a=='submit'),'123')

    def test_partial_array_submission_is_recorded_and_cannot_silently_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=settings(NS_LOG_ROOT=tmp,NS_PYTHON_BIN=sys.executable,
                LOOKBACKS_VALUES='32',HORIZON_VALUES='1',LAMBDAS_VALUES='1',CROSS_LOSSES_VALUES='bucket_ranknet')
            branches=ordinal.build_branches(s)
            def producer(s,b,action,dependency=''):
                if action=='submit':
                    error=RuntimeError('second array failed');error.job_ids=['123'];raise error
                return []
            with patch.object(ordinal.shutil,'which',return_value='/bin/true'),patch.object(ordinal,'run_producer',side_effect=producer):
                with self.assertRaises(RuntimeError): ordinal.train(s,branches)
                marker=next((Path(tmp)/'ordinal_submissions').glob('*.json'))
                self.assertEqual(json.loads(marker.read_text())['status'],'partial')
                with self.assertRaisesRegex(ValueError,'Submissao parcial'):
                    ordinal.train(s,branches)


if __name__=='__main__': unittest.main()
