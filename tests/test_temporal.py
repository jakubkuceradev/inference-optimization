from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from inference_baselines.temporal import current_mean5


class TemporalTests(unittest.TestCase):
    def test_feature_order_causality_and_interleaved_sources(self):
        index = np.array([[1, 7], [0, 4], [1, 5]])
        x = {0: np.arange(12*144, dtype=np.float32).reshape(12,144),
             1: np.arange(12*144, dtype=np.float32).reshape(12,144)/7}
        data = SimpleNamespace(index=index, x=x)
        result = current_mean5(data)
        for i,(f,e) in enumerate(index):
            np.testing.assert_array_equal(result[i,:144], x[f][e])
            np.testing.assert_array_equal(result[i,144:], x[f][e-4:e+1].astype(np.float64).mean(0).astype(np.float32))
        before = current_mean5(data, np.array([0]))
        x[1][8:] = 1e8
        np.testing.assert_array_equal(before, current_mean5(data, np.array([0])))
        with self.assertRaisesRegex(ValueError, "preceding"):
            current_mean5(SimpleNamespace(index=np.array([[0,3]]),x=x))

    def test_native_rolling_window_wraparound(self):
        source = Path(__file__).resolve().parents[1]/"src/inference_baselines/native/tree_runtime.cpp"
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/"runtime"
            subprocess.run(["c++","-O3","-std=c++17",str(source),"-ldl","-o",str(binary)],check=True,capture_output=True)
            subprocess.run([str(binary),"selftest"],check=True,capture_output=True)


if __name__ == "__main__":
    unittest.main()
