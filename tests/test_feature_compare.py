import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference_baselines.feature_compare import NATIVE, fit_ridge


class FeatureComparisonTests(unittest.TestCase):
    def test_native_recipes_match_independent_history(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(450, 144)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root/"features"
            subprocess.run(["g++", "-O2", "-std=c++17", "-ffp-contract=off", str(NATIVE/"feature_runtime.cpp"), "-ldl", "-o", str(binary)], check=True)
            np.save(root/"raw.npy", raw)
            offset = np.load(root/"raw.npy", mmap_mode="r").offset
            for recipe in ("current", "mean5", "current_mean5", "concat5", "multiscale384"):
                expected = []
                for edge in range(383, len(raw)):
                    mean = raw[edge-4:edge+1].astype(np.float64).mean(0).astype(np.float32)
                    if recipe == "current": row = raw[edge]
                    elif recipe == "mean5": row = mean
                    elif recipe == "current_mean5": row = np.concatenate((raw[edge], mean))
                    elif recipe == "concat5": row = raw[edge-np.arange(5)].ravel()
                    else:
                        row = np.concatenate([raw[edge], *[raw[edge-w+1:edge+1].astype(np.float64).mean(0).astype(np.float32) for w in (5, 20, 100, 384)], *[raw[edge-lag] for lag in (1, 4, 19, 99, 383)]])
                    expected.append(row)
                subprocess.run([str(binary), "features", recipe, str(root/"raw.npy"), str(offset), "383", "450", str(root/"out.f32"), "0"], check=True)
                np.testing.assert_array_equal(np.fromfile(root/"out.f32", np.float32).reshape(67, -1), expected)
                # Native benchmark rejects any mismatch between rolling and offline features.
                np.asarray(expected[1:], np.float32).tofile(root/"prepared.f32")
                raw[:384].tofile(root/"prefill.f32")
                raw[384:].tofile(root/"incoming.f32")
                np.zeros(3*len(expected[0])+3, np.float64).tofile(root/"ridge.f64")
                subprocess.run([str(binary), "run", recipe, "ridge", str(root/"ridge.f64"), "-",
                                str(root/"prepared.f32"), str(root/"timing.f64"), "stream",
                                str(root/"incoming.f32"), str(root/"prefill.f32"), "66"], check=True)

    def test_ridge_statistics_match_direct_weighted_solve(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(120, 8)).astype(np.float32)
        x[:, 7] = 0
        y = np.array([0]*20+[1]*60+[2]*40)
        vx = rng.normal(size=(50, 8)).astype(np.float32)
        vy = rng.integers(0, 3, 50)
        for weighting in ("uniform", "balanced"):
            coefficients, bias, _ = fit_ridge(x, y, vx, vy, {"weights": [weighting], "alphas": [10]})
            weights = np.ones(120) if weighting == "uniform" else 120/(3*np.bincount(y)[y])
            mean = np.average(x.astype(float), axis=0, weights=weights)
            scale = x.astype(float).std(0)
            scale[scale < 1e-12] = 1
            z = (x-mean)/scale
            target = 2*np.eye(3)[y]-1
            target_mean = np.average(target, axis=0, weights=weights)
            beta = np.linalg.solve(z.T@(weights[:, None]*z)+10*np.eye(8), z.T@(weights[:, None]*(target-target_mean)))
            np.testing.assert_allclose(coefficients, beta/scale[:, None], atol=1e-12)
            np.testing.assert_allclose(bias, target_mean-mean@coefficients, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
