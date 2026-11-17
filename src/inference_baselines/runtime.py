import os
from pathlib import Path
import time

import numpy as np
import torch


class Runtime:
    """Own setup and prepared-input invocation for one complete frozen model."""
    def __init__(self, model, name, output):
        output = Path(output)
        self.name = name
        self.provider = "PyTorch CPU"
        self.model = model
        started = time.perf_counter()
        if name == "compiled":
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output / "inductor-cache"))
            self.model = torch.compile(model, fullgraph=True, dynamic=False)
        elif name == "ort":
            import onnx
            import onnxruntime as ort
            path = output / "model.onnx"
            torch.onnx.export(model, torch.zeros(1, 384, 144), path, dynamo=False,
                              input_names=["input"], output_names=["logits"],
                              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
            onnx.checker.check_model(onnx.load(path))
            options = ort.SessionOptions()
            options.intra_op_num_threads = torch.get_num_threads()
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.optimized_model_filepath = str(output / "optimized.onnx")
            self.model = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            self.provider = self.model.get_providers()
        elif name != "eager":
            raise ValueError(name)
        self.setup_seconds = time.perf_counter()-started
        self.first_call_seconds = None

    def prepare(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        return x if self.name == "ort" else torch.from_numpy(x)

    def invoke(self, prepared):
        if self.name == "ort":
            return self.model.run(["logits"], {"input": prepared})[0]
        return self.model(prepared)

    def __call__(self, x):
        value = self.prepare(x)
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.invoke(value)
        if self.first_call_seconds is None:
            self.first_call_seconds = time.perf_counter()-started
        return result if isinstance(result, np.ndarray) else result.numpy()

    def metadata(self):
        return {"runtime": self.name, "provider": self.provider, "setup_seconds": self.setup_seconds,
                "first_call_seconds": self.first_call_seconds, "mode": "eval, inference_mode, FP32 CPU"}
