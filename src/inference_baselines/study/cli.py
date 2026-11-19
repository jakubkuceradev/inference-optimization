"""Thin coordinator; numerical work executes in isolated interpreter processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..common import configure, external, read_json, write_json


def worker(command, args, python=None):
    env=os.environ.copy()
    source=str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"]=source+os.pathsep+env.get("PYTHONPATH","")
    process=subprocess.Popen([python or sys.executable,"-m","inference_baselines.study.cli",command,*args],env=env)
    try: return process.wait()
    except KeyboardInterrupt:
        import signal
        process.send_signal(signal.SIGINT); process.wait(); raise


def train_campaign(root,device,resume,model,python):
    from .store import lock, study
    frozen=study(root)
    if device!=frozen["execution"]["training_device"]: raise ValueError("Frozen training device mismatch")
    with lock(Path(root)/"campaign.lock"):
        ledger_path=Path(root)/"campaign.json"
        ledger=read_json(ledger_path) if ledger_path.exists() else dict(elapsed=0.,status="pending")
        elapsed=ledger["elapsed"]
        if ledger["status"]=="running": elapsed+=max(0,time.time()-ledger["started_unix"])
        failures=[]
        for job in filtered(frozen,model):
            remaining=frozen["execution"]["campaign_seconds"]-elapsed
            if remaining<=0: break
            write_json(ledger_path,dict(status="running",elapsed=elapsed,started_unix=time.time(),job=job["key"]))
            started=time.perf_counter()
            args=["--study",str(root),"--device",device,"--model",job["config"]["id"],"--campaign-remaining",str(remaining)]
            if resume: args.append("--resume")
            code=worker("_fit",args,python)
            elapsed+=time.perf_counter()-started
            write_json(ledger_path,dict(status="idle",elapsed=elapsed))
            if code: failures.append(job["config"]["id"])
        write_json(ledger_path,dict(status="failed" if failures else "budget-stopped" if elapsed>=frozen["execution"]["campaign_seconds"] else "idle",elapsed=elapsed,failures=failures))
        if failures: raise RuntimeError(f"Failed training jobs: {failures}")


def filtered(frozen,model):
    jobs=[j for j in frozen["jobs"] if model is None or j["config"]["id"]==model]
    if not jobs: raise ValueError("Unknown declared model filter")
    return jobs


def main(argv=None):
    parser=argparse.ArgumentParser(prog="inference-study")
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare"); p.add_argument("--development-source",required=True); p.add_argument("--source-manifest"); p.add_argument("--output",required=True)
    p=commands.add_parser("synthetic"); p.add_argument("--output",required=True)
    p=commands.add_parser("plan"); p.add_argument("--config",required=True); p.add_argument("--data")
    p=commands.add_parser("preflight"); p.add_argument("--config",required=True); p.add_argument("--data",required=True); p.add_argument("--output",required=True); p.add_argument("--device",choices=("cpu","cuda"),required=True); p.add_argument("--seconds",type=float,default=60); p.add_argument("--python")
    p=commands.add_parser("freeze"); p.add_argument("--config",required=True); p.add_argument("--data",required=True); p.add_argument("--preflight",required=True); p.add_argument("--execution",required=True); p.add_argument("--output",required=True); p.add_argument("--python")
    for name in ("train","evaluate","benchmark","profile","_fit"):
        p=commands.add_parser(name); p.add_argument("--study",required=True); p.add_argument("--device",choices=("cpu","cuda"),required=True); p.add_argument("--model"); p.add_argument("--python"); p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
        if name in ("train","_fit"): p.add_argument("--resume",action="store_true")
        if name=="_fit": p.add_argument("--campaign-remaining",type=float,required=True)
        if name in ("evaluate","benchmark"): p.add_argument("--ort-include")
        if name=="benchmark": p.add_argument("--cpu",type=int,required=True)
        if name=="profile": p.add_argument("--calls",type=int,default=100)
    p=commands.add_parser("report"); p.add_argument("--study",required=True); p.add_argument("--output",required=True); p.add_argument("--plots",action="store_true")
    args=parser.parse_args(argv)
    cmd=args.command
    # Named interpreter applies to preflight, freeze and each numerical worker.
    if cmd in ("preflight","freeze") and args.python:
        original=list(argv if argv is not None else sys.argv[1:]); i=original.index("--python"); original[i:i+2]=[]
        code=worker(original[0],original[1:],args.python)
        if code: raise SystemExit(code)
        return
    if cmd in ("preflight","_fit","evaluate","benchmark","profile"):
        configure(1,getattr(args,"cpu",None))
    from .data import Development, prepare, synthetic
    from .plan import describe, load
    from .store import freeze, study
    if cmd=="prepare": result=prepare(args.development_source,args.output,args.source_manifest)
    elif cmd=="synthetic": result=synthetic(args.output)
    elif cmd=="plan":
        plan=load(args.config); result=dict(state="draft",plan=plan,jobs=describe(plan))
        if args.data: result["data_hash"]=Development(args.data).binding
    elif cmd=="preflight":
        from .preflight import preflight
        result=preflight(load(args.config),args.data,args.output,args.device,args.seconds)
    elif cmd=="freeze": result=freeze(load(args.config),args.data,read_json(Path(args.preflight)/"preflight.json"),read_json(args.execution),args.output)
    elif cmd=="train":
        train_campaign(args.study,args.device,args.resume,args.model,args.python); result={"study":args.study}
    elif cmd=="report":
        from .report import report
        result=report(args.study,args.output,args.plots)
    elif cmd=="_fit":
        from .training import fit
        frozen=study(args.study)
        result=fit(frozen,filtered(frozen,args.model)[0],args.device,args.resume,campaign_remaining=args.campaign_remaining)
    else:
        if not args.worker:
            frozen=study(args.study)
            for job in filtered(frozen,args.model):
                arguments=["--study",args.study,"--device",args.device,"--model",job["config"]["id"],"--worker"]
                if cmd=="benchmark": arguments += ["--cpu",str(args.cpu)]
                if cmd=="profile": arguments += ["--calls",str(args.calls)]
                if getattr(args,"ort_include",None): arguments += ["--ort-include",args.ort_include]
                code=worker(cmd,arguments,args.python)
                if code: raise SystemExit(code)
            return
        from .evidence import evaluate,benchmark,profile
        frozen=study(args.study); job=filtered(frozen,args.model)[0]
        if cmd=="evaluate": result=evaluate(frozen,job,args.device,args.ort_include)
        elif cmd=="benchmark": result=benchmark(frozen,job,args.device,args.ort_include)
        else:
            if args.calls<=0: raise ValueError("Positive profile call count required")
            result=profile(frozen,job,args.device,args.calls)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__": main()
