import copy
import multiprocessing
import os
import subprocess
import sys
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from inference_baselines.common import digest, read_json, write_json
from inference_baselines.model import MLPLOB
from inference_baselines.study.data import Development, synthetic
from inference_baselines.study.plan import compose, describe, load, model_config, validate
from inference_baselines.study.preflight import preflight
from inference_baselines.study.store import activity, freeze, job_path, provenance
from inference_baselines.study.training import fit, load_bundle

CONFIGS=Path(__file__).parents[1]/'src/inference_baselines/configs'


def contend(queue):
    try:
        with activity(measurement=True): queue.put('acquired')
    except RuntimeError: queue.put('locked')


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def setup_study(self,root):
        synthetic(root/'data')
        plan=load(CONFIGS/'mlplob-smoke.json')
        evidence=preflight(plan,root/'data',root/'preflight','cpu',seconds=5)
        settings=dict(batch_size=8,batch_exceptions={},run_seconds=60,campaign_seconds=300,artifact_store=str(root/'store'),training_device='cpu')
        frozen=freeze(plan,root/'data',evidence,settings,root/'study')
        return frozen,frozen['jobs'][0]

    def test_explicit_union_and_expansion_pairs(self):
        limited=load(CONFIGS/'mlplob-limited.json'); medium=load(CONFIGS/'mlplob-medium.json')
        self.assertEqual(len(compose([limited,medium])),47)
        self.assertEqual(len(limited['jobs']),44)
        self.assertEqual(len(medium['jobs']),3)
        self.assertTrue(all(j['history']<384 for j in limited['jobs']))
        self.assertEqual(sum(j['history']==384 for j in medium['jobs']),1)
        reduced=[j for j in limited['jobs'] if j['expansion']=='reduced']
        self.assertEqual(len(reduced),8)
        for j in reduced:
            pair=[k for k in limited['jobs'] if all(k[x]==j[x] for x in ('history','width','depth','features','seed')) and k['expansion']=='standard']
            self.assertEqual(len(pair),1)
            self.assertLess(sum(p.numel() for p in MLPLOB(**model_config(j)).parameters()),sum(p.numel() for p in MLPLOB(**model_config(pair[0])).parameters()))
        for mutation in ('history','unknown','duplicate','horizon','reference'):
            bad=copy.deepcopy(limited)
            if mutation=='history': bad['jobs'][0]['history']=388
            elif mutation=='unknown': bad['jobs'][0]['mystery']=1
            elif mutation=='duplicate': bad['jobs'].append(bad['jobs'][0])
            elif mutation=='horizon': bad['recipe']['horizon']=20
            else: bad['jobs'][0].update(history=384,width=32)
            with self.assertRaises(ValueError): validate(bad)

    def test_default_model_is_standard_and_aliases_count_once(self):
        kwargs=dict(hidden_dim=32,num_layers=3,seq_size=32,num_features=40,dataset_type='FI_2010')
        a=MLPLOB(**kwargs).eval(); b=MLPLOB(**kwargs,intermediate_expansion=4,final_expansion=2).eval()
        b.load_state_dict(a.state_dict()); x=torch.randn(2,32,40)
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
        self.assertEqual(a.first_layer.weight.data_ptr(),a.layers[0].weight.data_ptr())
        self.assertGreater(sum(v.numel() for v in a.state_dict().values()),sum(v.numel() for v in a.parameters()))

    def test_corpus_rejects_heldout_hash_boundary_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); synthetic(root/'data'); data=root/'data'
            c=Development(data)
            np.testing.assert_array_equal(c.windows('train',[0],16,40),c.windows('train',[0],16,144)[:,:,:40])
            manifest=read_json(data/'manifest.json')
            bad=copy.deepcopy(manifest); bad['segments']['validation'][0]['day']=8
            with self.assertRaises(ValueError): Development(data,bad)
            bad=copy.deepcopy(manifest); bad['support']=128
            with self.assertRaises(ValueError): Development(data,bad)
            bad=copy.deepcopy(manifest); bad['files']['train-raw.npy']='bad'
            with self.assertRaises(ValueError): Development(data,bad)
            ids=np.load(data/'train-index.npy'); ids[0,2]-=1; np.save(data/'train-index.npy',ids)
            bad=copy.deepcopy(manifest); bad['files']['train-index.npy']=digest(data/'train-index.npy')
            with self.assertRaises(ValueError): Development(data,bad)
            targets=np.load(data/'validation-targets.npy'); targets[0]=3; np.save(data/'validation-targets.npy',targets)
            bad=copy.deepcopy(manifest); bad['files']['validation-targets.npy']=digest(data/'validation-targets.npy')
            with self.assertRaises(ValueError): Development(data,bad)

    def test_resume_matches_uninterrupted_and_interrupted_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen,job=self.setup_study(root)
            result=fit(frozen,job,'cpu',stop_after=2)
            self.assertEqual(result['status'],'interrupted')
            snapshot=load_bundle(job_path(frozen,job))
            fit(frozen,job,'cpu',resume=True,stop_after=4)
            resumed=load_bundle(job_path(frozen,job))
            alternate=copy.deepcopy(frozen); alternate['execution']['artifact_store']=str(root/'fresh')
            fit(alternate,job,'cpu',stop_after=4)
            straight=load_bundle(job_path(alternate,job))
            for k in straight['current']:
                torch.testing.assert_close(straight['current'][k],resumed['current'][k],rtol=0,atol=0)
            self.assertEqual(straight['best_epoch'],resumed['best_epoch'])
            self.assertEqual(straight['patience'],resumed['patience'])
            for a,b in zip(straight['history'],resumed['history']):
                self.assertEqual(a['validation'],b['validation']); self.assertEqual(a['train_loss'],b['train_loss'])
            interrupted=copy.deepcopy(frozen); interrupted['execution']['artifact_store']=str(root/'fault')
            def fault(phase,epoch):
                if phase=='before-publication' and epoch==3: raise RuntimeError('injected crash')
            with self.assertRaises(RuntimeError): fit(interrupted,job,'cpu',fault=fault)
            self.assertEqual(load_bundle(job_path(interrupted,job))['epoch'],2)
            fit(interrupted,job,'cpu',resume=True,stop_after=4)
            recovered=load_bundle(job_path(interrupted,job))
            for k in resumed['current']:
                torch.testing.assert_close(recovered['current'][k],resumed['current'][k],rtol=0,atol=0)
            changed=copy.deepcopy(job); changed['binding']['recipe']['learning_rate']=.003
            with self.assertRaises(ValueError): fit(frozen,changed,'cpu',resume=True)

    def test_host_measurement_lock_excludes_other_store(self):
        context=multiprocessing.get_context('spawn'); queue=context.Queue()
        with activity():
            process=context.Process(target=contend,args=(queue,)); process.start(); process.join(15)
            self.assertEqual(process.exitcode,0); self.assertEqual(queue.get(timeout=2),'locked')
        process=context.Process(target=contend,args=(queue,)); process.start(); process.join(15)
        self.assertEqual(process.exitcode,0); self.assertEqual(queue.get(timeout=2),'acquired')
        with tempfile.TemporaryDirectory() as tmp, activity():
            env=dict(os.environ,TMPDIR=tmp)
            code="from inference_baselines.study.store import activity;\nwith activity(measurement=True): pass"
            result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Runner activity is locked',result.stderr)

    def test_preflight_records_allocation_failure(self):
        class AllocationFailure(torch.nn.Module):
            def __init__(self,**kwargs):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.ones(1))
            def to(self,*args,**kwargs):
                raise torch.OutOfMemoryError('injected allocation failure')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); synthetic(root/'data')
            with patch('inference_baselines.study.preflight.MLPLOB',AllocationFailure):
                evidence=preflight(load(CONFIGS/'mlplob-smoke.json'),root/'data',root/'preflight','cpu',5)
            self.assertEqual(evidence['status'],'preflight-recorded')
            self.assertEqual(len(evidence['measurements']),2)
            self.assertTrue(all(m['status']=='unsupported' for m in evidence['measurements']))

    def test_hardware_identity_excludes_dynamic_clock(self):
        from inference_baselines.common import hardware_identity
        first='model name : Test CPU\nvendor_id : Test\ncpu MHz : 1000.0\n'
        second=first.replace('1000.0','4200.0')
        with patch('inference_baselines.common.Path.read_text',side_effect=[first,second]):
            self.assertEqual(hardware_identity(),hardware_identity())

    def test_cli_pins_all_threads_before_numerical_imports(self):
        code = """
import os,sys
from pathlib import Path
from inference_baselines.study.cli import main
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
cpu=min(os.sched_getaffinity(0))
try: main(['benchmark','--study','/tmp/nonexistent-mlplob-study-affinity','--device','cpu','--cpu',str(cpu),'--worker'])
except FileNotFoundError: pass
for task in Path('/proc/self/task').iterdir():
    try: assert os.sched_getaffinity(int(task.name)) == {cpu}
    except ProcessLookupError: pass
"""
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

    def test_budget_stop_without_epoch_is_terminal_and_preserves_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen,job=self.setup_study(root)
            stopped=fit(frozen,job,'cpu',campaign_remaining=0)
            self.assertEqual(stopped['status'],'budget-stopped')
            self.assertEqual(stopped['epochs'],0)
            again=fit(frozen,job,'cpu',resume=True)
            self.assertEqual(again,stopped)
            self.assertFalse((job_path(frozen,job)/'selected.json').exists())

    def test_equal_score_earliest_selection_and_eight_epoch_minimum(self):
        from inference_baselines.study.training import quality as actual_quality
        def constant_quality(targets,logits,ids):
            result=actual_quality(targets,logits,ids)
            result['equal_day_macro_f1']=.5
            return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen,job=self.setup_study(root)
            with patch('inference_baselines.study.training.quality',side_effect=constant_quality):
                stopped=fit(frozen,job,'cpu')
            state=load_bundle(job_path(frozen,job))
            self.assertEqual(stopped['status'],'early-stopped')
            self.assertEqual(state['epoch'],8)
            self.assertEqual(state['best_epoch'],1)
            self.assertEqual(state['patience'],7)

    def test_freeze_rejects_stale_preflight_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen,job=self.setup_study(root)
            settings=frozen['execution']; settings['batch_size']=7
            with self.assertRaises(ValueError): freeze(frozen['plan'],root/'data',frozen['preflight'],settings,root/'bad')
            settings['batch_size']=8; bad=copy.deepcopy(frozen['preflight']); bad['code']['git_revision']='stale'
            with self.assertRaises(ValueError): freeze(frozen['plan'],root/'data',bad,settings,root/'bad')


if __name__=='__main__': unittest.main()
