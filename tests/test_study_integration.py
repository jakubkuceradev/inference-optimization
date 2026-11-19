"""Actual adapters; skipped boundaries are never claimed as integrated."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from inference_baselines.common import read_json
from inference_baselines.study.data import synthetic
from inference_baselines.study.evidence import benchmark,evaluate,profile,canonical
from inference_baselines.study.plan import load
from inference_baselines.study.preflight import preflight
from inference_baselines.study.report import report
from inference_baselines.study.store import freeze,job_path
from inference_baselines.study.training import fit,selected

CONFIG=Path(__file__).parents[1]/'src/inference_baselines/configs/mlplob-smoke.json'


class StudyIntegration(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.temp=tempfile.TemporaryDirectory(prefix='mlplob-integration-test-')
        self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        synthetic(self.root/'data')
        plan=load(CONFIG)
        evidence=preflight(plan,self.root/'data',self.root/'preflight','cpu',5)
        self.frozen=freeze(plan,self.root/'data',evidence,dict(batch_size=8,batch_exceptions={},run_seconds=60,campaign_seconds=300,artifact_store=str(self.root/'store'),training_device='cpu'),self.root/'study')
        self.job=self.frozen['jobs'][0]
        result=fit(self.frozen,self.job,'cpu')
        self.assertIn(result['status'],('completed','early-stopped'))

    @unittest.skipUnless(os.environ.get('ORT_INCLUDE_DIR'),'Official ORT headers not configured')
    def test_native_validation_actual_timed_outputs_and_report(self):
        ev=evaluate(self.frozen,self.job,'cpu')
        self.assertEqual(ev['status'],'completed')
        self.assertEqual(ev['runtime']['runtime'],'native-ort')
        self.assertEqual(ev['runtime']['shape'],(1,4,40))
        bench=benchmark(self.frozen,self.job,'cpu')
        self.assertEqual(bench['status'],'completed')
        self.assertEqual(len(bench['records']),4)
        root=job_path(self.frozen,self.job)/'cpu'
        for row in bench['records']:
            with np.load(root/row['file']) as values:
                self.assertEqual(values['logits'].shape,(8,3))
                self.assertTrue((values['ns']>0).all())
                np.testing.assert_array_equal(values['decisions'],values['logits'].argmax(1))
        self.assertEqual(profile(self.frozen,self.job,'cpu',5)['status'],'completed')
        r=report(self.root/'study',self.root/'report')
        self.assertEqual(len(r['rows']),4)
        self.assertFalse(any(row['verified_frontier'] for row in r['rows']))
        self.assertTrue(all(row['evaluation_status']=='unmeasured' for row in r['rows'] if row['device']=='cuda'))

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA unavailable; graph integration unverified')
    def test_real_cuda_graph_roundtrip(self):
        evaluate(self.frozen,self.job,'cpu')  # Publishes the single CPU oracle even without native headers.
        ev=evaluate(self.frozen,self.job,'cuda')
        self.assertEqual(ev['status'],'completed')
        self.assertEqual(ev['runtime']['runtime'],'cuda-graph')
        bench=benchmark(self.frozen,self.job,'cuda')
        self.assertEqual(bench['status'],'completed')
        self.assertEqual(len(bench['records']),4)

    def test_oracle_is_published_once_and_reused(self):
        from unittest.mock import patch
        from inference_baselines.study.data import Development
        root=job_path(self.frozen,self.job)
        model,checkpoint=selected(root,self.job)
        corpus=Development(self.frozen['data'])
        first,metadata=canonical(self.frozen,self.job,model,checkpoint,corpus,create=True)
        with patch('inference_baselines.study.evidence.predict',side_effect=AssertionError('must reuse CPU oracle')):
            second,again=canonical(self.frozen,self.job,model,checkpoint,corpus)
        self.assertEqual(metadata,again)
        np.testing.assert_array_equal(first['reference'],second['reference'])
        np.testing.assert_array_equal(first['timed_reference'],second['timed_reference'])

    def test_explicit_missing_cuda_status(self):
        if torch.cuda.is_available(): self.skipTest('CUDA available')
        ev=evaluate(self.frozen,self.job,'cuda')
        self.assertEqual(ev['status'],'unsupported')
        self.assertEqual(benchmark(self.frozen,self.job,'cuda')['status'],'unsupported')
        r=report(self.root/'study',self.root/'report')
        self.assertTrue(all(row['benchmark_status']=='unsupported' for row in r['rows'] if row['device']=='cuda'))


if __name__=='__main__': unittest.main()
