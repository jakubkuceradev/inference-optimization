import copy
import importlib.util
import os
import sys
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from inference_baselines.bin import BiN
from inference_baselines.common import digest, read_json, write_json
from inference_baselines.data import Corpus, eligible
from inference_baselines.metrics import compare, latency_summary, score
from inference_baselines.model import MLPLOB
from inference_baselines.runtime import Runtime
from inference_baselines.training import ema_update, save_torch
from inference_baselines.timing import measurement_order


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_partition_history_and_future_exclusions(self):
        edges = eligible(203800, 254750)
        self.assertEqual((edges[0], edges[-1], len(edges)), (204183, 254739, 50557))
        self.assertEqual(len(eligible(0, 393)), 0)
        self.assertEqual(eligible(100, 494).tolist(), [483])

    def test_timing_plan_is_fixed_and_balanced(self):
        names = ["eager", "compiled", "ort"]
        first = measurement_order(names, 3)
        self.assertEqual(first, measurement_order(names, 3))
        expected = {(n,b) for n in names for b in ("prepared","local_preparation")}
        for repeat in first:
            self.assertEqual({tuple(x) for x in repeat}, expected)

    def test_tiny_train_select_reload_and_full_evaluation(self):
        from inference_baselines.training import train, load_selected
        from inference_baselines.evaluation import evaluate
        from inference_baselines.data import PROTOCOL
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/"data"
            data.mkdir()
            rng = np.random.default_rng(9)
            np.save(data/"x-0.npy",rng.normal(size=(1200,144)).astype(np.float32))
            np.save(data/"y-0.npy",np.arange(1200)%3)
            for partition, edges in (("train",[383,384,385,386]),("validation",[783,784]),("test",[1183,1184])):
                np.save(data/f"{partition}-index.npy",np.column_stack([np.zeros(len(edges),dtype=np.int64),edges]))
            write_json(data/"manifest.json",{"protocol":PROTOCOL,"sources":[{"file_id":0,"sha256":"test-only","member":"synthetic"}],
                "arrays":{p.name:digest(p) for p in data.glob("*.npy")}})
            write_json(data/"status.json",{"status":"completed"})
            config = read_json(Path(__file__).resolve().parents[1]/"src/inference_baselines/configs/fi2010.json")
            config.update({"epochs":1,"batch_size":2,"evaluation_batch_size":2,"seeds":[42]})
            config["model"].update({"hidden_dim":8,"num_layers":1})
            train(data,root/"smoke-42",config,42,smoke=True)
            status = train(data,root/"seed-42",config,42)
            self.assertEqual(status["status"],"completed")
            self.assertTrue(read_json(root/"seed-42/reload.json")["exact"])
            model,manifest,selected = load_selected(root/"seed-42")
            result = evaluate(data,root/"seed-42",root/"evaluation-42-eager","eager")
            self.assertEqual(result["reference"]["count"],2)
            self.assertEqual(result["numerical"]["tolerance_pass_count"],2)
            self.assertEqual(result["checkpoint_sha256"],selected["checkpoint_sha256"])
            # Exercise real timing/report plumbing on tiny synthetic inputs only.
            from inference_baselines.timing import benchmark
            from inference_baselines.report import report
            affinity = os.sched_getaffinity(0)
            try:
                os.sched_setaffinity(0,{min(affinity)})
                timing_config = {**config,"calls":3,"repeats":2,"warmup":1}
                benchmark(data,root/"seed-42",root/"measurements-42","eager",timing_config)
            finally:
                os.sched_setaffinity(0,affinity)
            timed = read_json(root/"measurements-42/eager/result.json")
            self.assertEqual(len(timed["observations"]),4)
            for observation in timed["observations"]:
                self.assertEqual(observation["count"],3)
                path = root/"measurements-42/eager"/f"{observation['boundary']}-{observation['repeat']}-ns.npy"
                self.assertEqual(len(np.load(path)),3)
            summary = report(root,root/"report-v1")
            self.assertEqual(summary["aggregate"]["completed_seeds"],1)
            self.assertEqual(summary["aggregate"]["scheduled_seeds"],3)
            with self.assertRaisesRegex(ValueError,"Smoke checkpoints"):
                evaluate(data,root/"smoke-42",root/"invalid-evaluation","eager")

    def test_lazy_window_alignment_ids_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.arange(400*144, dtype=np.float32).reshape(400,144)
            np.save(root/"x-0.npy", x)
            np.save(root/"y-0.npy", np.arange(400)%3)
            np.save(root/"test-index.npy", [[0,383],[0,384]])
            write_json(root/"manifest.json", {"sources": [{"file_id": 0,"sha256":"hash","member":"matrix"}], "arrays":{p.name:digest(p) for p in root.glob("*.npy")}})
            write_json(root/"status.json", {"status":"completed"})
            data = Corpus(root, "test")
            np.testing.assert_array_equal(data[0][0], x[:384])
            self.assertEqual(data[1][1], 0)
            self.assertIn(":384:test:file-0-test:internal-unverified", data.ids()[1])
            np.save(root/"y-0.npy", np.zeros(400))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Corpus(root,"test")

    def test_metrics_support_majority_and_visible_failures(self):
        result = score(np.array([0,1,2,2]), np.array([2,2,2,2]))
        self.assertEqual(result["support"], [1,1,2])
        self.assertEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 2/9)
        a = np.array([[1.,2.,3.],[np.nan,0.,1.]], dtype=np.float32)
        b = np.array([[1.1,2.,3.],[np.nan,0.,1.]], dtype=np.float32)
        check = compare(a,b,3e-5,3e-4)
        self.assertEqual(check["tolerance_pass_count"], 0)
        self.assertEqual(check["class_disagreements_finite"], 0)
        self.assertEqual(check["reference_nonfinite"], 1)
        self.assertEqual(latency_summary([1000,2000,3000])["p50_us"], 2.)

    def test_ema_visits_aliased_parameter_once(self):
        model = MLPLOB(8,1,8,4,"FI_2010")
        average = copy.deepcopy(model)
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(2)
            for p in average.parameters():
                p.zero_()
        ema_update(average, model, .5)
        torch.testing.assert_close(average.first_layer.weight, torch.ones_like(average.first_layer.weight))

    def test_bin_repair_preserves_identity_and_constant_nonfinites(self):
        layer = BiN(4,8)
        identity = id(layer.y1)
        with torch.no_grad():
            layer.y1.fill_(-1)
        layer(torch.randn(2,4,8))
        self.assertEqual(id(layer.y1), identity)
        self.assertEqual(layer.repairs, 1)
        self.assertAlmostEqual(layer.y1.item(), .01)
        self.assertFalse(torch.isfinite(layer(torch.zeros(1,4,8))).all())

    def test_checkpoint_rng_optimizer_roundtrip(self):
        torch.manual_seed(42)
        model = MLPLOB(8,1,8,4,"FI_2010")
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        x = torch.randn(2,8,4)
        def step(m, opt):
            opt.zero_grad()
            m(x).sum().backward()
            opt.step()
        step(model, optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"resume.pt"
            save_torch(path,{"model":model.state_dict(),"optimizer":optimizer.state_dict(),"rng":torch.get_rng_state()})
            restored = MLPLOB(8,1,8,4,"FI_2010")
            opt = torch.optim.Adam(restored.parameters())
            state = torch.load(path, weights_only=True)
            restored.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            step(model,optimizer)
            step(restored,opt)
            for p,q in zip(model.parameters(),restored.parameters(),strict=True):
                torch.testing.assert_close(p,q,atol=0,rtol=0)
            torch.set_rng_state(state["rng"])
            first = torch.rand(3)
            torch.set_rng_state(state["rng"])
            torch.testing.assert_close(first,torch.rand(3),atol=0,rtol=0)

    def test_pinned_upstream_positive_coefficient_equivalence(self):
        source = Path("/home/ashen/.local/share/dev/inference-optimization/research/mlp-focused-2026-11-12/models/TLOB")
        if not source.exists():
            self.skipTest("Pinned source snapshot not installed; run on the evidence host")
        provenance = read_json(Path(__file__).resolve().parents[1]/"src/inference_baselines/configs/source.json")
        for name, expected in provenance["files_sha256"].items():
            self.assertEqual(digest(source/name), expected, name)
        # Load only the isolated upstream model files, no application/training code.
        saved = {k:sys.modules.get(k) for k in ("constants","models","models.bin")}
        try:
            sys.modules["constants"] = types.SimpleNamespace(DEVICE="cpu")
            sys.modules["models"] = types.ModuleType("models")
            for name, file in (("models.bin","bin.py"),("upstream_mlplob","mlplob.py")):
                spec = importlib.util.spec_from_file_location(name, source/"models"/file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            original = module.MLPLOB(144,3,384,144,"FI_2010").eval()
            adapted = MLPLOB(144,3,384,144,"FI_2010").eval()
            adapted.load_state_dict(original.state_dict(), strict=True)
            with torch.inference_mode():
                x = torch.randn(2,384,144)
                torch.testing.assert_close(original(x), adapted(x), atol=0,rtol=0)
                for axis in (1,2):
                    x = torch.randn(1,384,144).mean(axis,keepdim=True).expand(1,384,144)
                    torch.testing.assert_close(original(x),adapted(x),atol=0,rtol=0,equal_nan=True)
        finally:
            for key,value in saved.items():
                if value is None:
                    sys.modules.pop(key,None)
                else:
                    sys.modules[key] = value
            sys.modules.pop("upstream_mlplob",None)


if __name__ == "__main__":
    unittest.main()
