"""Model discovery reads source declarations without importing model packages."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.ordinal_model_registry import discover_models,resolve_model


def write(root,relative,text):
    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)


class OrdinalModelRegistryTest(unittest.TestCase):
    def test_real_native_exports_qualified_paths_and_generated_factories(self):
        registry=discover_models()
        for requested,label,adapter in [
            ('PatchTST','PatchTST','transformer_adapter'),
            ('time_series_library.PatchTST','PatchTST','transformer_adapter'),
            ('ts_benchmark.baselines.time_series_library.models.PatchTST.PatchTST','PatchTST','transformer_adapter'),
            ('DTAF','DTAF',None),('SparseTSF','SparseTSF',None),
            ('ts_benchmark.baselines.duet.duet.DUET','DUET',None),
            ('darts.ARIMA','ARIMA',None),('ARIMA','ARIMA',None),
            ('ts_benchmark.baselines.darts.darts_models.ARIMA','ARIMA',None)]:
            with self.subTest(requested=requested):
                name,spec=resolve_model(requested,registry=registry)
                self.assertEqual(name,label)
                self.assertEqual(spec['adapter'],adapter)
                self.assertTrue((ROOT/spec['source_path']).is_file())
        self.assertNotIn('ts_benchmark.baselines.darts.darts_deep_model_adapter',registry)

    def test_discovery_does_not_import_torch_or_other_optional_dependencies(self):
        code='''import sys
from scripts.ordinal_model_registry import discover_models, resolve_model
r=discover_models(); assert resolve_model("PatchTST",registry=r)[0]=="PatchTST"
assert not {"torch","numpy","pandas","darts","merlion"}.intersection(sys.modules)
'''
        result=subprocess.run([sys.executable,'-S','-c',code],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_new_native_export_is_discovered_without_named_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root,'ts_benchmark/baselines/future/__init__.py','__all__=["FutureNet","make_future"]\nfrom .network import FutureNet, make_future\n')
            write(root,'ts_benchmark/baselines/future/network.py','import unavailable_package\nclass FutureNet:\n    pass\ndef make_future(**kwargs):\n    pass\nCONFIG={}\n')
            label,spec=resolve_model('FutureNet',root)
            self.assertEqual(label,'FutureNet')
            self.assertEqual(spec['model_name'],'future.FutureNet')
            self.assertEqual(resolve_model('make_future',root)[0],'make_future')
            self.assertEqual(resolve_model('future.network.CONFIG',root)[0],'future.network.CONFIG')
            self.assertEqual(resolve_model('future.network.make_future',root)[0],'make_future')

    def test_raw_torch_class_does_not_assume_transformer_forward_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write(root,'ts_benchmark/baselines/raw/__init__.py','__all__=["Net"]\nfrom .network import Net\n')
            write(root,'ts_benchmark/baselines/raw/network.py','import torch.nn as nn\nclass Net(nn.Module):\n    pass\n')
            name,spec=resolve_model('raw.Net',root)
            self.assertEqual(name,'Net')
            self.assertIsNone(spec['adapter'])

    def test_ambiguous_short_name_requires_explicit_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for package in ('one','two'):
                write(root,f'ts_benchmark/baselines/{package}/__init__.py','__all__=["Twin"]\nfrom .model import Twin\n')
                write(root,f'ts_benchmark/baselines/{package}/model.py','class Twin:\n    pass\n')
            with self.assertRaisesRegex(ValueError,'ambiguo.*one.Twin.*two.Twin'):
                resolve_model('Twin',root)
            self.assertEqual(resolve_model('one.Twin',root)[0],'one.Twin')
            self.assertEqual(resolve_model('two.Twin',root)[0],'two.Twin')

    def test_invalid_or_missing_model_paths_fail_without_interpretation(self):
        for name in ['../DUET','duet..DUET','/tmp/DUET','DUET;echo','DUET()', 'not_a_model']:
            with self.subTest(name=name),self.assertRaises(ValueError):resolve_model(name)


if __name__=='__main__':unittest.main()
