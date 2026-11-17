import argparse
from pathlib import Path
import json

from .common import configure, read_json


def main():
    parser = argparse.ArgumentParser(description="FI-2010 same-weight reference baselines")
    parser.add_argument("command", choices=["audit", "prepare", "control", "smoke", "train", "diagnose", "evaluate", "benchmark", "report"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent/"configs/fi2010.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true", help="Explicitly exceed the smoke feasibility budget")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Training device; inference is always CPU")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--runtime", choices=["eager", "compiled", "ort", "all"], default="eager")
    args = parser.parse_args()
    if args.runtime == "all" and args.command != "benchmark":
        parser.error("--runtime all is supported only for benchmark")
    requirements = {"control": ["data"], "smoke": ["data"], "train": ["data"],
                    "diagnose": ["data", "run"], "evaluate": ["data", "run"],
                    "benchmark": ["data", "run"], "report": ["run"]}
    for name in requirements.get(args.command, []):
        if getattr(args, name) is None:
            parser.error(f"--{name} is required for {args.command}")
    config = read_json(args.config)
    configure(1 if args.command in ("diagnose", "benchmark") else config["threads"], args.cpu)
    if args.command in ("prepare", "audit"):
        if args.source is None:
            parser.error("--source is required")
        from .data import prepare
        result = prepare(args.source, args.output)
    elif args.command in ("smoke", "train"):
        if args.data is None:
            parser.error("--data is required")
        from .training import train
        result = train(args.data, args.output, config, args.seed, smoke=args.command == "smoke", resume=args.resume, allow_long_run=args.allow_long_run, device=args.device)
    elif args.command == "control":
        from .common import external
        from .evaluation import majority
        output = external(args.output)
        output.mkdir(parents=True, exist_ok=False)
        result = majority(args.data, output)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.data, args.run, args.output, args.runtime)
    elif args.command == "diagnose":
        from .evaluation import diagnose
        result = diagnose(args.data, args.run, args.output, args.runtime)
    elif args.command == "benchmark":
        from .timing import benchmark
        result = benchmark(args.data, args.run, args.output, args.runtime, config)
    else:
        from .report import report
        result = report(args.run, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
