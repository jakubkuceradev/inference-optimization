from pathlib import Path

import numpy as np

from .common import digest, external, read_json, write_json
from .data import LIMITS


def report(root, output):
    root, output = Path(root).resolve(), external(output)
    output.mkdir(parents=True, exist_ok=False)
    schedule = read_json(root / "schedule.json") if (root/"schedule.json").exists() else {"seeds": [42, 43, 44]}
    runs, quality, timings = {}, [], []
    for seed in schedule["seeds"]:
        path = root / f"seed-{seed}"
        runs[str(seed)] = read_json(path/"status.json") if (path/"status.json").exists() else {"status": "not_run", "reason": schedule.get("reason", "not executed")}
        for name in ("eager", "compiled", "ort"):
            for prefix, destination in (("evaluation", quality), ("benchmark", timings)):
                path = root / f"{prefix}-{seed}-{name}"
                if prefix == "benchmark" and (root/f"measurements-{seed}"/name).exists():
                    path = root/f"measurements-{seed}"/name
                status = read_json(path/"status.json") if (path/"status.json").exists() else {"status": "not_run", "reason": "requires completed selected checkpoint and prior stages"}
                item = {"seed": seed, "runtime": name, "status": status, "artifact": str(path)}
                if status["status"] == "completed":
                    item["result"] = read_json(path/"result.json")
                    runtime_metadata = item["result"].get("runtime_setup", item["result"].get("runtime"))
                    if runtime_metadata["runtime"] != name or (prefix == "evaluation" and item["result"]["seed"] != seed):
                        raise ValueError(f"Report runtime/seed identity mismatch: {path}")
                    selected = read_json(root/f"seed-{seed}"/"selected.json")
                    training_manifest = read_json(root/f"seed-{seed}"/"manifest.json")
                    if item["result"]["checkpoint_sha256"] != selected["checkpoint_sha256"] or digest(root/f"seed-{seed}"/"selected.pt") != selected["checkpoint_sha256"] or item["result"]["dataset_manifest_sha256"] != training_manifest["dataset_manifest_sha256"]:
                        raise ValueError(f"Report refuses mixed checkpoint/data results: {path}")
                destination.append(item)
    # Aggregate only matched eager reference results, never the best test seed.
    references = [i["result"]["reference"] for i in quality if i["runtime"] == "eager" and "result" in i and "macro_f1" in i["result"]["reference"]]
    aggregate = {"completed_seeds": len(references), "scheduled_seeds": len(schedule["seeds"])}
    for metric in ("macro_f1", "accuracy", "log_loss"):
        if references:
            values = [r[metric] for r in references]
            aggregate[metric] = {"mean": float(np.mean(values)), "std_population": float(np.std(values)) if len(values)>1 else None, "values": values}
    valid = []
    for item in timings:
        if "result" not in item:
            continue
        r = item["result"]
        n = r["numerical"]
        diagnostics = [v for v in r["diagnostics"].values() if isinstance(v, dict) and "tolerance_pass_count" in v]
        if n["tolerance_pass_count"] == n["count"] and all(d["tolerance_pass_count"] == d["finite_row_count"] and d["nonfinite_mask_disagreements"] == 0 for d in diagnostics):
            medians = [o["p50_us"] for o in r["observations"] if o["boundary"] == "prepared"]
            valid.append((float(np.median(medians)), item["runtime"]))
    strongest = min(valid)[1] if valid else None
    control = read_json(root/"control-v1/majority.json") if (root/"control-v1/majority.json").exists() else None
    diagnostics = {name: read_json(path) for name in ("eager", "compiled", "ort")
                   if (path := root/f"diagnostic-42-{name}"/"result.json").exists()}
    result = {"schedule": schedule, "training": runs, "quality": quality, "timings": timings,
              "aggregate": aggregate, "strongest_validated_runtime": strongest, "limitations": LIMITS,
              "majority_control": control, "tooling_diagnostics": diagnostics}
    result["runtime_selection_criterion"] = "lowest median repeat prepared-input p50 among paths passing finite full-test and diagnostic gates, with matching nonfinite masks"
    write_json(output / "report.json", result)
    lines = ["# FI-2010 baseline report", "", "Protocol: fi2010-source-compatible-caveated-v1.", "",
             "## Training", ""]
    for seed, status in runs.items():
        lines.append(f"- Seed {seed}: {status['status']}. {status.get('reason', '')}")
        selected_path = root/f"seed-{seed}"/"selected.json"
        if selected_path.exists():
            selected = read_json(selected_path)
            lines.append(f"  Selected EMA epoch {selected['epoch']} (zero based), validation loss {selected['validation_loss']:.6f}, checkpoint SHA256 {selected['checkpoint_sha256']}.")
    if schedule.get("deferred_seeds"):
        lines += ["", f"Deferred by user: {schedule['deferred_seeds']}."]
    if len(references)<2:
        lines += ["", "Training variability is not estimated from a single completed seed."]
    if (root/"smoke-42/status.json").exists():
        smoke = read_json(root/"smoke-42/status.json")
        lines += ["", "Smoke feasibility (diagnostic weights, not a trained reference):", "", f"```json\n{__import__('json').dumps(smoke, indent=2)}\n```"]
    lines += ["", "## Quality and numerical checks", ""]
    for item in quality:
        r = item.get("result")
        text = f"Seed {item['seed']} / {item['runtime']}: {item['status']['status']}"
        if r:
            text += f"; reference {r['reference']}; candidate {r['candidate']}; numerical {r['numerical']}; checkpoint {r['checkpoint_sha256']}"
        lines.append(f"- {text}. Artifacts: {item['artifact']}")
    lines += ["", f"Seed aggregate: {aggregate}", "", "## Performance", "",
              f"Strongest applicable validated runtime: {strongest or 'not established; no qualifying matched measurements'}.", ""]
    for item in timings:
        lines.append(f"- Seed {item['seed']} / {item['runtime']}: {item['status']['status']}. Artifacts: {item['artifact']}")
    completed_timings = [item for item in timings if "result" in item]
    if completed_timings:
        lines += ["", "Repeat quantiles below are medians across repeats, not pooled quantiles. Maximum includes every retained observation.", "",
                  "| Runtime | Boundary | Calls | p50 µs | p95 µs | p99 µs | Max µs | Service calls/s | Wall calls/s |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        setup_lines = []
        for item in completed_timings:
            timing = item["result"]
            for boundary in ("prepared", "local_preparation"):
                rows = [row for row in timing["observations"] if row["boundary"] == boundary]
                med = lambda name: float(np.median([row[name] for row in rows]))
                lines.append(f"| {item['runtime']} | {boundary} | {sum(row['count'] for row in rows)} | {med('p50_us'):.2f} | {med('p95_us'):.2f} | {med('p99_us'):.2f} | {max(row['maximum_us'] for row in rows):.2f} | {med('service_calls_per_second'):.2f} | {med('wall_calls_per_second'):.2f} |")
            setup = timing["runtime_setup"]
            setup_lines.append(f"{item['runtime']} setup {setup['setup_seconds']:.3f} s; first call {setup['first_call_seconds']:.3f} s; checkpoint loading {timing['checkpoint_load_seconds']:.3f} s; peak process RSS {timing['peak_process_rss_kib']/1024:.1f} MiB; checkpoint {timing['checkpoint_bytes']} bytes.")
        lines += [""] + setup_lines
        lines += ["", "Peak RSS includes all resident runtimes and compiler setup, not isolated per-runtime memory. Numerical failures remain visible above and do not establish equivalent speedups.",
                  "Prepared timing starts with resident inputs; local preparation includes history assembly/layout conversion. Both consume output argmax. Disk/decompression, raw engineered-feature generation, network and queueing are excluded."]
    if control:
        lines += ["", f"Majority control on all eligible test targets: {control['metrics']}."]
    if diagnostics:
        lines += ["", "## Tooling diagnostics (smoke weights, validation/synthetic only)", ""]
        for name, diagnostic in diagnostics.items():
            checks = diagnostic["checks"]
            lines.append(f"- {name}: validation bulk {checks['validation_bulk']['tolerance_pass_count']}/64; batch one {checks['validation_batch_one']['tolerance_pass_count']}/64; synthetic {checks['diagnostic_bulk']}. These are not trained-baseline quality or latency results.")
    lines += ["", "## Limits", ""] + [f"- {limit}" for limit in LIMITS]
    lines += ["- Overlapping windows are dependent observations; no independent-sample confidence claim.",
              "- No profitability, broad market generalization, novel optimization, or deployment SLA claim."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    return {"report": str(output/"report.md"), "aggregate": aggregate, "strongest_validated_runtime": strongest}
