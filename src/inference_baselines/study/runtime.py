"""Static native ORT and CUDA Graph adapters with completed-call boundaries."""
import ctypes
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np
import torch

from ..common import digest


class Unsupported(RuntimeError):
    pass


def export(model, shape, directory):
    import onnx
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path = directory/"model.onnx"
    torch.onnx.export(model,torch.randn(shape),path,dynamo=False,input_names=["input"],output_names=["logits"],opset_version=18)
    onnx.checker.check_model(onnx.load(path))
    return path


class NativeORT:
    name = "native-ort"
    def __init__(self, model, config, directory, include=None):
        import onnxruntime as ort
        self.shape = (1,config["history"],config["features"])
        directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
        include = Path(include or os.environ.get("ORT_INCLUDE_DIR", "/nonexistent"))
        if not (include/"onnxruntime_cxx_api.h").exists():
            raise Unsupported("Native ORT requires ORT_INCLUDE_DIR or --ort-include (matching official ORT headers)")
        libraries = list((Path(ort.__file__).parent/"capi").glob("libonnxruntime.so*"))
        if not libraries: raise Unsupported("ONNX Runtime shared library unavailable")
        library = sorted(libraries,key=lambda p:len(p.name))[-1].resolve()
        path = export(model,self.shape,directory)
        source = Path(__file__).parents[1]/"native/study_ort.cpp"
        target = directory/"study-ort.so"
        command = [*shlex.split(os.environ.get("CXX","c++")),"-std=c++17","-O3","-shared","-fPIC",str(source),"-I",str(include),str(library),f"-Wl,-rpath,{library.parent}","-o",str(target)]
        result = subprocess.run(command,text=True,capture_output=True)
        if result.returncode: raise Unsupported(f"Native ORT build failed: {result.stderr}")
        self.ort_library = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        self.lib = ctypes.CDLL(str(target))
        self.lib.study_error.restype = ctypes.c_char_p
        self.lib.study_ort_version.restype = ctypes.c_char_p
        self.lib.study_create.argtypes = [ctypes.c_char_p,ctypes.c_int64,ctypes.c_int64]
        self.lib.study_create.restype = ctypes.c_void_p
        self.lib.study_destroy.argtypes = [ctypes.c_void_p]
        ptr = ctypes.POINTER(ctypes.c_float)
        self.lib.study_call.argtypes = [ctypes.c_void_p,ptr,ptr,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
        self.lib.study_call.restype = ctypes.c_double
        self.handle = self.lib.study_create(os.fsencode(path),self.shape[1],self.shape[2])
        if not self.handle: raise RuntimeError(self.lib.study_error().decode())
        version = self.lib.study_ort_version().decode()
        if version != ort.__version__: raise RuntimeError("Loaded native ORT version mismatch")
        mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libonnxruntime.so' in line})
        if len(mapped) != 1 or Path(mapped[0]).resolve() != library:
            raise RuntimeError(f"Unexpected loaded ORT libraries: {mapped}")
        self.output = np.empty((1,3),np.float32)
        self.output_ptr = self.output.ctypes.data_as(ptr)
        self.decision = ctypes.c_int()
        self.metadata = dict(runtime=self.name,provider="CPUExecutionProvider",library=str(library),library_sha256=digest(library),
                             version=version,compiler_command=command,compiler_version=subprocess.run([command[0],"--version"],capture_output=True,text=True).stdout,
                             source_sha256=digest(source),headers={p.name:digest(p) for p in sorted(include.glob('onnxruntime*.h'))},
                             model_sha256=digest(path),binary_sha256=digest(target),threads=1,shape=self.shape,
                             timing="native steady_clock; audit copying and Python calls outside boundary")

    def call(self, x, boundary="primary"):
        if x.shape != self.shape or x.dtype != np.float32 or not x.flags.c_contiguous: raise ValueError("Static contiguous FP32 window required")
        ns = self.lib.study_call(self.handle,x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),self.output_ptr,boundary=="prepared",ctypes.byref(self.decision))
        if ns < 0: raise RuntimeError(self.lib.study_error().decode())
        return ns, self.output.copy(), self.decision.value

    def close(self):
        if getattr(self,"handle",None): self.lib.study_destroy(self.handle); self.handle = None

    def __del__(self): self.close()


class CudaGraph:
    name = "cuda-graph"
    def __init__(self,model,config,directory=None,include=None):
        if not torch.cuda.is_available(): raise Unsupported("CUDA hardware/runtime unavailable")
        self.shape = (1,config["history"],config["features"])
        self.model = model.to("cuda").eval()
        self.host = torch.empty(self.shape,dtype=torch.float32,pin_memory=True)
        self.host.normal_()
        self.input = self.host.to("cuda")
        self.output_host = torch.empty((1,3),dtype=torch.float32,pin_memory=True)
        try:
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.inference_mode():
                for _ in range(3): self.model(self.input)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph),torch.inference_mode(): self.output = self.model(self.input)
            torch.cuda.synchronize()
        except RuntimeError as error:
            raise Unsupported(f"CUDA capture unsupported: {error}") from error
        self.metadata = dict(runtime=self.name,shape=self.shape,device=torch.cuda.get_device_name(0),
                             cuda=torch.version.cuda,torch=torch.__version__,tf32=False,
                             buffers="stable pinned host input/output; static device input/output",threads=torch.get_num_threads())

    def call(self,x,boundary="primary"):
        if x.shape != self.shape or x.dtype != np.float32 or not x.flags.c_contiguous: raise ValueError("Static contiguous FP32 window required")
        source = torch.from_numpy(x)
        with torch.inference_mode():
            if boundary == "prepared":
                self.host.copy_(source); self.input.copy_(self.host,non_blocking=True); torch.cuda.synchronize()
            start = time.perf_counter_ns()
            if boundary == "primary":
                self.host.copy_(source); self.input.copy_(self.host,non_blocking=True)
            self.graph.replay()
            self.output_host.copy_(self.output,non_blocking=True)
            torch.cuda.current_stream().synchronize()
            decision = int(self.output_host.numpy().argmax())
            ns = time.perf_counter_ns()-start
        return ns,self.output_host.numpy().copy(),decision

    def close(self): pass


class EagerCPU:
    name = "eager-cpu"
    def __init__(self,model,config,directory=None,include=None):
        self.model=model.cpu().eval()
        self.shape=(1,config["history"],config["features"])
        self.input=torch.empty(self.shape,dtype=torch.float32)
        self.metadata=dict(runtime=self.name,shape=self.shape,torch=torch.__version__,threads=torch.get_num_threads(),provider="PyTorch CPU")

    def call(self,x,boundary="primary"):
        source=torch.from_numpy(x)
        with torch.inference_mode():
            if boundary=="prepared": self.input.copy_(source)
            start=time.perf_counter_ns()
            if boundary=="primary": self.input.copy_(source)
            output=self.model(self.input)
            decision=int(output.argmax())
            ns=time.perf_counter_ns()-start
        return ns,output.numpy().copy(),decision

    def close(self): pass


def adapter(device,model,config,directory,include=None,runtime=None):
    if runtime=="eager-cpu": return EagerCPU(model,config,directory,include)
    return (NativeORT if device=="cpu" else CudaGraph)(model,config,directory,include)


def select_runtime(device,model,config,directory,corpus,include=None):
    import copy
    candidate=adapter(device,copy.deepcopy(model),config,directory,include)
    if device=="cuda":
        return candidate,dict(chosen=candidate.name,evidence="Established explicit graph replay removes repeated eager launch overhead on this static graph",source="wiki/research/baseline-models.md",comparison="No redundant eager CUDA latency series")
    eager=EagerCPU(copy.deepcopy(model),config)
    windows=[corpus.windows("validation",[i%len(corpus.arrays["validation"]["index"])],config["history"],config["features"]) for i in range(24)]
    paths=[candidate,eager]; observations={p.name:[] for p in paths}
    for x in windows[:4]:
        for path in paths: path.call(x)
    for i,x in enumerate(windows[4:]):
        for path in paths[::1 if i%2==0 else -1]: observations[path.name].append(path.call(x)[0])
    chosen=min(paths,key=lambda p:np.median(observations[p.name]))
    evidence=dict(chosen=chosen.name,boundary="primary",calls=20,warmups=4,durations_ns=observations,
                  evidence="Bounded interleaved native ORT / eager CPU completed-call comparison; median selects path",input_positions=list(range(4,24)))
    for path in paths:
        if path is not chosen: path.close()
    return chosen,evidence


def fingerprint(metadata):
    """Execution identity excludes artifact paths and timing observations."""
    from .plan import identity
    keys=("runtime","provider","shape","torch","cuda","device","tf32","threads",
          "library_sha256","version","source_sha256","headers","model_sha256","binary_sha256")
    return identity({k:metadata[k] for k in keys if k in metadata})
