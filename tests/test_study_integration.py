"""Actual adapters; skipped boundaries are never claimed as integrated."""
import copy
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
from inference_baselines.study.training import fit,selected,load_bundle

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
        # Runtime smoke needs a real fitted checkpoint, not an unbounded test of
        # convergence on synthetic labels. Stopping-policy tests cover that rule.
        result=fit(self.frozen,self.job,'cpu',stop_after=2)
        self.assertEqual(result['status'],'interrupted')

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

    @unittest.skipUnless(torch.cuda.is_available(),'Actual CUDA unavailable; training resume unverified')
    def test_cuda_scheduler_optimizer_and_rng_resume(self):
        from unittest.mock import patch
        from inference_baselines.study.training import quality as actual_quality
        evidence=preflight(self.frozen['plan'],self.root/'data',self.root/'cuda-preflight','cuda',5)
        settings=dict(self.frozen['execution'],training_device='cuda',artifact_store=str(self.root/'cuda-store'))
        frozen=freeze(self.frozen['plan'],self.root/'data',evidence,settings,self.root/'cuda-study')
        job=frozen['jobs'][0]
        def metrics(targets,logits,ids):
            result=actual_quality(targets,logits,ids)
            result['equal_day_macro_f1']=.5;result['pooled']['log_loss']=.7
            return result
        with patch('inference_baselines.study.training.quality',side_effect=metrics):
            fit(frozen,job,'cuda',stop_after=3)
            fit(frozen,job,'cuda',resume=True,stop_after=4)
            fresh=copy.deepcopy(frozen);fresh['execution']['artifact_store']=str(self.root/'cuda-fresh')
            fit(fresh,job,'cuda',stop_after=4)
        resumed=load_bundle(job_path(frozen,job));straight=load_bundle(job_path(fresh,job))
        self.assertAlmostEqual(resumed['optimizer']['param_groups'][0]['lr'],.00015)
        self.assertEqual(straight['scheduler'],resumed['scheduler'])
        self.assertEqual(straight['patience'],resumed['patience'])
        self.assertEqual(straight['best_validation_loss_epoch'],resumed['best_validation_loss_epoch'])
        for key in straight['current']:
            torch.testing.assert_close(straight['current'][key],resumed['current'][key],rtol=0,atol=0)
        for a,b in zip(straight['cuda_rng'],resumed['cuda_rng'],strict=True):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

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
