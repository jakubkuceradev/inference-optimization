from pathlib import Path
import tempfile
import unittest

import numpy as np

from inference_baselines.common import digest, read_json, write_json
from inference_baselines.shallow import matrix, Predictor, ridge_fit, snapshot


class ShallowTests(unittest.TestCase):
    def test_snapshot_matrix_preserves_interleaved_identity(self):
        from types import SimpleNamespace
        class Data(SimpleNamespace):
            def __len__(self):
                return len(self.index)
        data = Data(index=np.array([[1, 2], [0, 1], [1, 0]]),
                    x={0: np.zeros((4,144), np.float32), 1: np.ones((4,144), np.float32)})
        value = matrix(data)
        np.testing.assert_array_equal(value[:, 0], [1, 0, 1])
        np.testing.assert_array_equal(value[0:1], snapshot(data, 0))
        value[0, 0] = 2
        self.assertEqual(data.x[1][2, 0], 1)

    def test_ridge_relabeling_and_serialized_predictor(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(40, 5)).astype(np.float32)
        y = np.arange(40)%3
        weight, bias = ridge_fit(x, y, 1.)
        permuted = np.array([2, 0, 1])
        weight2, bias2 = ridge_fit(x, permuted[y], 1.)
        np.testing.assert_allclose((x@weight2+bias2)[:, permuted], x@weight+bias, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory)/"ridge.npz", weight=weight, bias=bias)
            predictor = Predictor(directory, {"kind": "ridge", "file": "ridge.npz"})
            np.testing.assert_array_equal(predictor(x), x@weight+bias)


if __name__ == "__main__":
    unittest.main()
