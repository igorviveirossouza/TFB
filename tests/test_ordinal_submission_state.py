"""Prediction-scoped job ownership, conservative legacy migration and diagnostics."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import ordinal_experiment as ordinal


def small_settings(root, **overrides):
    env=dict(NS_LOG_ROOT=str(root/'logs'),PRED_ROOT=str(root/'predictions'),
        NS_RESULT_ROOT=str(root/'results'),NS_PYTHON_BIN=sys.executable,
        LOOKBACKS_VALUES='32',HORIZON_VALUES='1',LAMBDAS_VALUES='1',
        CROSS_LOSSES_VALUES='bucket_ranknet')
    env.update({k:str(v) for k,v in overrides.items()})
    return ordinal.settings(env)


def quiet_train(s, branches, producer):
    with patch.object(ordinal.shutil,'which',return_value='/bin/true'),patch.object(ordinal,'run_producer',side_effect=producer):
        with contextlib.redirect_stdout(io.StringIO()):ordinal.train(s,branches)


def successful_producer(calls):
    next_id=123
    def run(s,b,action,dependency=''):
        nonlocal next_id
        calls.append((action,dependency))
        if action=='submit':
            result=[str(next_id)];next_id+=1;return result
        return []
    return run


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class OrdinalSubmissionStateTest(unittest.TestCase):
    def test_separate_prediction_roots_can_share_logs_without_false_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            first=small_settings(root,PRED_ROOT=root/'first',MODEL='DUET',TRAIN_VALUES='90')
            second=small_settings(root,PRED_ROOT=root/'second',MODEL='Nonstationary_Transformer',TRAIN_VALUES='80')
            quiet_train(first,ordinal.build_branches(first),successful_producer(calls))
            quiet_train(second,ordinal.build_branches(second),successful_producer(calls))
            self.assertEqual(sum(action=='submit' for action,_ in calls),2)
            for s in (first,second):
                marker=next(ordinal.submission_state(s).glob('*.json'))
                stored=json.loads(marker.read_text())
                self.assertEqual(stored['schema_version'],2)
                self.assertEqual(stored['pred_root'],s['PRED_ROOT'])
                self.assertEqual(stored['log_root'],s['NS_LOG_ROOT'])
                self.assertEqual(stored['job_ids'],['123'])
                self.assertIn('config',stored)

    def test_changing_logs_cannot_bypass_same_output_configuration_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            first=small_settings(root,MODEL='DUET')
            quiet_train(first,ordinal.build_branches(first),successful_producer(calls))
            changed=small_settings(root,MODEL='PatchTST',NS_LOG_ROOT=root/'other_logs')
            calls.clear()
            with self.assertRaisesRegex(ValueError,'mesmo destino.*IDs Slurm: 123.*campos alterados'):
                quiet_train(changed,ordinal.build_branches(changed),successful_producer(calls))
            self.assertFalse(calls)
            info=ordinal.submission_status(changed,ordinal.build_branches(changed))
            self.assertTrue(info['branches'][0]['config_changed'])
            self.assertIn('models',[v['field'] for v in info['branches'][0]['changed_fields']])

    def test_same_output_same_config_with_new_logs_skips_existing_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            first=small_settings(root)
            quiet_train(first,ordinal.build_branches(first),successful_producer(calls))
            changed=small_settings(root,NS_LOG_ROOT=root/'other_logs')
            calls.clear()
            quiet_train(changed,ordinal.build_branches(changed),successful_producer(calls))
            self.assertFalse(any(action=='submit' for action,_ in calls))

    def test_legacy_without_destination_stays_ambiguous_even_with_same_prepared_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root);branches=ordinal.build_branches(s);branch=branches[0]
            legacy=Path(s['NS_LOG_ROOT'])/'ordinal_submissions'/ordinal.marker_name(branch)
            write_json(legacy,dict(signature=branch['signature'],job_ids=['321'],branch=branch['relative'],status='submitted'))
            # A design prepared in this root does not prove those job IDs target it.
            prepared=Path(s['PRED_ROOT'])/branch['relative']/'treino=80/experimento=composto/run_config.json'
            write_json(prepared,dict(config=branch['config']))
            calls=[]
            with self.assertRaisesRegex(ValueError,'Destino do registro legado nao identificado.*321'):
                quiet_train(s,branches,successful_producer(calls))
            self.assertFalse(calls)
            self.assertFalse((ordinal.submission_state(s)/ordinal.marker_name(branch)).exists())
            info=ordinal.submission_status(s,branches)['branches'][0]
            self.assertEqual(info['legacy_disposition'],'destination_unidentified')
            self.assertEqual(info['status'],'legacy_ambiguous')
            self.assertEqual(info['legacy_job_ids'],['321'])
            self.assertIn('nao comprova',info['evidence_note'])
            self.assertTrue(legacy.exists())

    def test_legacy_with_explicit_matching_destination_migrates_without_resubmission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root);branches=ordinal.build_branches(s);branch=branches[0]
            legacy=Path(s['NS_LOG_ROOT'])/'ordinal_submissions'/ordinal.marker_name(branch)
            write_json(legacy,ordinal.marker_record(s,branch,'submitted',['321']))
            calls=[]
            quiet_train(s,branches,successful_producer(calls))
            self.assertFalse(any(action=='submit' for action,_ in calls))
            migrated=json.loads((ordinal.submission_state(s)/ordinal.marker_name(branch)).read_text())
            self.assertEqual(migrated['migrated_from'],str(legacy))
            self.assertEqual(migrated['job_ids'],['321'])
            self.assertTrue(legacy.exists())

    def test_explicit_legacy_ownership_persists_even_if_current_config_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=small_settings(root,MODEL='DUET');branch=ordinal.build_branches(old)[0]
            legacy=Path(old['NS_LOG_ROOT'])/'ordinal_submissions'/ordinal.marker_name(branch)
            write_json(legacy,ordinal.marker_record(old,branch,'submitted',['321']))
            current=small_settings(root,MODEL='PatchTST');calls=[]
            with self.assertRaisesRegex(ValueError,'Configuracao alterada'):
                quiet_train(current,ordinal.build_branches(current),successful_producer(calls))
            self.assertFalse(calls)
            self.assertTrue((ordinal.submission_state(current)/ordinal.marker_name(branch)).exists())
            current['NS_LOG_ROOT']=str(root/'yet_another_log_root')
            with self.assertRaisesRegex(ValueError,'Configuracao alterada'):
                quiet_train(current,ordinal.build_branches(current),successful_producer(calls))

    def test_explicit_legacy_other_destination_does_not_block_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=small_settings(root,PRED_ROOT=root/'old');branch=ordinal.build_branches(old)[0]
            legacy=Path(old['NS_LOG_ROOT'])/'ordinal_submissions'/ordinal.marker_name(branch)
            write_json(legacy,ordinal.marker_record(old,branch,'submitted',['321']))
            current=small_settings(root,PRED_ROOT=root/'new',MODEL='DUET');calls=[]
            quiet_train(current,ordinal.build_branches(current),successful_producer(calls))
            self.assertEqual(sum(action=='submit' for action,_ in calls),1)
            self.assertTrue(legacy.exists())

    def test_read_only_status_creates_no_paths_and_accesses_no_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root)
            with patch.object(ordinal,'run_producer',side_effect=AssertionError('must not submit')):
                info=ordinal.submission_status(s,ordinal.build_branches(s))
            self.assertEqual(info['branches'][0]['status'],'not_submitted')
            self.assertEqual(list(root.iterdir()),[])
            result=subprocess.run([sys.executable,str(ROOT/'scripts/ordinal_experiment.py'),'--submission-status'],
                env={**os.environ,**s},cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['pred_root'],s['PRED_ROOT'])
            self.assertEqual(list(root.iterdir()),[])

    def test_missing_job_ids_keep_intent_and_block_automatic_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root);branches=ordinal.build_branches(s);calls=[]
            def no_ids(s,b,action,dependency=''):
                calls.append(action);return []
            with self.assertRaisesRegex(RuntimeError,'sem IDs Slurm'):
                quiet_train(s,branches,no_ids)
            marker=next(ordinal.submission_state(s).glob('*.json'))
            self.assertEqual(json.loads(marker.read_text())['status'],'submitting')
            calls.clear()
            with self.assertRaisesRegex(ValueError,'sem confirmacao final'):
                quiet_train(s,branches,no_ids)
            self.assertFalse(calls)

    def test_interleaved_existing_branches_do_not_create_parallel_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root,CROSS_LOSSES_VALUES='ordinal_ce bucket_ranknet bucket_hinge')
            branches=ordinal.build_branches(s);calls=[]
            quiet_train(s,[branches[0],branches[2]],successful_producer(calls))
            s['NS_RESUME_CHAIN_OLD_JOBS']='1';calls.clear()
            with self.assertRaisesRegex(ValueError,'intercalados'):
                quiet_train(s,branches,successful_producer(calls))
            self.assertFalse(calls)

    def test_parallel_history_cannot_be_assumed_serial_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root,CROSS_LOSSES_VALUES='ordinal_ce bucket_ranknet',NS_PARALLEL_BRANCHES='1')
            branches=ordinal.build_branches(s);calls=[]
            quiet_train(s,branches[:1],successful_producer(calls))
            s.update(NS_PARALLEL_BRANCHES='0',NS_RESUME_CHAIN_OLD_JOBS='1');calls.clear()
            with self.assertRaisesRegex(ValueError,'Historico de submissao paralelo'):
                quiet_train(s,branches,successful_producer(calls))
            self.assertFalse(calls)

    def test_unselected_history_blocks_new_serial_branch_and_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=small_settings(root,CROSS_LOSSES_VALUES='ordinal_ce');calls=[]
            quiet_train(old,ordinal.build_branches(old),successful_producer(calls))
            new=small_settings(root,CROSS_LOSSES_VALUES='bucket_hinge');branches=ordinal.build_branches(new);calls.clear()
            with self.assertRaisesRegex(ValueError,'ramos nao selecionados.*123'):
                quiet_train(new,branches,successful_producer(calls))
            self.assertFalse(calls)
            self.assertEqual(ordinal.submission_status(new,branches)['unselected_history'][0]['job_ids'],['123'])

    def test_reordered_existing_prefix_does_not_bypass_serial_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root,CROSS_LOSSES_VALUES='ordinal_ce bucket_ranknet bucket_hinge')
            branches=ordinal.build_branches(s);calls=[]
            quiet_train(s,branches[:2],successful_producer(calls))
            s['NS_RESUME_CHAIN_OLD_JOBS']='1';calls.clear()
            with self.assertRaisesRegex(ValueError,'nao corresponde ao encadeamento'):
                quiet_train(s,[branches[1],branches[0],branches[2]],successful_producer(calls))
            self.assertFalse(calls)

    def test_new_external_dependency_is_not_silently_ignored_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=small_settings(root,CROSS_LOSSES_VALUES='ordinal_ce bucket_ranknet')
            branches=ordinal.build_branches(s);calls=[]
            quiet_train(s,branches[:1],successful_producer(calls))
            s.update(NS_RESUME_CHAIN_OLD_JOBS='1',NS_AFTER_JOB='999');calls.clear()
            with self.assertRaisesRegex(ValueError,'NS_AFTER_JOB nao pertence'):
                quiet_train(s,branches,successful_producer(calls))
            self.assertFalse(calls)

    def test_unselected_legacy_history_cannot_be_evaded_by_changing_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=small_settings(root,CROSS_LOSSES_VALUES='ordinal_ce');branch=ordinal.build_branches(old)[0]
            legacy=Path(old['NS_LOG_ROOT'])/'ordinal_submissions'/ordinal.marker_name(branch)
            write_json(legacy,dict(signature=branch['signature'],branch=branch['relative'],status='submitted',job_ids=['321']))
            current=small_settings(root,CROSS_LOSSES_VALUES='bucket_listnet');branches=ordinal.build_branches(current);calls=[]
            with self.assertRaisesRegex(ValueError,'ramos nao selecionados.*321'):
                quiet_train(current,branches,successful_producer(calls))
            self.assertFalse(calls)
            history=ordinal.submission_status(current,branches)['unselected_history']
            self.assertEqual(history[0]['status'],'legacy_ambiguous')
            self.assertEqual(history[0]['storage'],'legacy')

    def test_relative_paths_resolve_against_checkout_even_from_outside_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            env={**os.environ,'PRED_ROOT':'relative_pred','NS_LOG_ROOT':'relative_log',
                 'NS_RESULT_ROOT':'relative_result','LOOKBACKS_VALUES':'32','HORIZON_VALUES':'1',
                 'LAMBDAS_VALUES':'1','CROSS_LOSSES_VALUES':'bucket_ranknet'}
            result=subprocess.run([sys.executable,str(ROOT/'scripts/ordinal_experiment.py'),'--submission-status'],
                cwd=root,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            info=json.loads(result.stdout)
            self.assertEqual(info['pred_root'],str(ROOT/'relative_pred'))
            self.assertEqual(info['log_root'],str(ROOT/'relative_log'))
            self.assertEqual(info['result_root'],str(ROOT/'relative_result'))
            self.assertEqual(info['state_root'],str(ROOT/'relative_pred/.ordinal_submissions'))
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()
