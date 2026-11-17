import unittest

import numpy as np
import torch

from inference_baselines.gpu import CudaRuntime
from inference_baselines.model import MLPLOB


@unittest.skipUnless(torch.cuda.is_available(), "CUDA integration test requires a visible device")
class GpuTests(unittest.TestCase):
    def test_graph_updates_inputs_and_preserves_batch_profiles(self):
        torch.manual_seed(1729)
        model = MLPLOB(8, 1, 8, 4, "FI_2010").cuda().eval()
        eager, graph = CudaRuntime(model, "eager"), CudaRuntime(model, "graph")
        rng = np.random.default_rng(42)
        for batch in (1, 2, 1, 2):
            x = rng.normal(size=(batch, 8, 4)).astype(np.float32)
            np.testing.assert_allclose(graph(x), eager(x), atol=3e-5, rtol=3e-4)
        self.assertEqual(set(graph.profiles), {(1, 8, 4), (2, 8, 4)})


if __name__ == "__main__":
    unittest.main()
