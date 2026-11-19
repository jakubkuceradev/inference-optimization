"""Read-only evidence aggregation; no model or data execution."""
import csv
import json
from pathlib import Path

from ..common import digest, external, read_json, write_json
from .store import job_path, study


def report(study_root, output, plots=False):
    frozen=study(study_root,check_data=False)
    output=external(output); output.mkdir(parents=True,exist_ok=True)
    rows=[]; artifacts=[]
    for job in frozen["jobs"]:
        directory=job_path(frozen,job)
        training=read_json(directory/"training.json") if (directory/"training.json").exists() else {"status":"pending"}
        if training.get("binding",job["binding"])!=job["binding"]: raise ValueError("Changed training binding")
        selected=read_json(directory/"selected.json") if (directory/"selected.json").exists() else None
        if selected and digest(directory/"selected.pt")!=selected["sha256"]: raise ValueError("Changed selected checkpoint")
        for device in frozen["plan"]["devices"]:
            root=directory/device
            ev=read_json(root/"evaluation.json") if (root/"evaluation.json").exists() else {"status":"unmeasured"}
            bench=read_json(root/"benchmark.json") if (root/"benchmark.json").exists() else {"status":"unmeasured"}
            for evidence in (ev,bench):
                bound=evidence.get("binding")
                if bound and (bound["job_key"]!=job["key"] or bound["data_hash"]!=frozen["data_hash"] or not selected or bound["checkpoint_sha256"]!=selected["sha256"] or bound["device"]!=device):
                    raise ValueError("Stale report evidence")
            if ev["status"]=="completed" and digest(root/"validation.npz")!=ev["predictions_sha256"]: raise ValueError("Changed predictions")
            if bench["status"]=="completed":
                if ev["status"]!="completed" or bench["binding"]!=ev["binding"]: raise ValueError("Benchmark/evaluation binding mismatch")
                for record in bench["records"]:
                    if digest(root/record["file"])!=record["sha256"]: raise ValueError("Changed timing evidence")
            artifacts.append(dict(model=job["config"],unique_parameters=job["unique_parameters"],device=device,
                                  training=training,evaluation=ev,benchmark=bench,selected=selected))
            hardware=json.dumps(bench.get("environment",ev.get("environment",{})).get("hardware",{}).get(device,"unmeasured"),sort_keys=True)
            quality=ev.get("quality")
            for boundary in ("primary","prepared"):
                summary=bench.get("summary",{}).get(boundary,{})
                rows.append(dict(model=job["config"]["id"],job_key=job["key"],seed=job["config"]["seed"],features=job["config"]["features"],
                    unique_parameters=job["unique_parameters"],device=device,runtime=bench.get("runtime",ev.get("runtime",{})).get("runtime","unmeasured"),hardware=hardware,
                    boundary=boundary,kind=frozen["plan"]["kind"],training_status=training["status"],evaluation_status=ev["status"],benchmark_status=bench["status"],
                    numerical_status=bench.get("numerical_status",ev.get("numerical_status","unmeasured")),
                    equal_day_macro_f1=quality["equal_day_macro_f1"] if quality else None,
                    median_repeat_p50_us=summary.get("median_repeat_p50_us"),worst_repeat_p99_us=summary.get("worst_repeat_p99_us"),
                    checkpoint_sha256=selected["sha256"] if selected else None,verified_frontier=False))
    # Frontiers are meaningful only within one host/device/runtime/boundary/seed.
    for row in rows:
        if row["kind"]=="smoke" or row["numerical_status"]!="passed" or row["benchmark_status"]!="completed" or row["equal_day_macro_f1"] is None: continue
        group=(row["hardware"],row["device"],row["runtime"],row["boundary"],row["seed"])
        peers=[r for r in rows if (r["hardware"],r["device"],r["runtime"],r["boundary"],r["seed"])==group and r["numerical_status"]=="passed" and r["benchmark_status"]=="completed" and r["equal_day_macro_f1"] is not None]
        row["verified_frontier"]=not any(r["worst_repeat_p99_us"]<=row["worst_repeat_p99_us"] and r["equal_day_macro_f1"]>=row["equal_day_macro_f1"] and (r["worst_repeat_p99_us"]<row["worst_repeat_p99_us"] or r["equal_day_macro_f1"]>row["equal_day_macro_f1"]) for r in peers)
    result=dict(plan=frozen["plan"],study_manifest_sha256=digest(Path(study_root)/"study.json"),rows=rows,artifacts=artifacts,
                limitations="Validation study; separate seeds, hardware and timing boundaries. Smoke points excluded. Failed logit gates remain diagnostic. No confidence intervals from one seed.")
    write_json(output/"report.json",result)
    with (output/"report.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups={}
        for row in rows:
            if row["worst_repeat_p99_us"] is None or row["equal_day_macro_f1"] is None: continue
            groups.setdefault((row["hardware"],row["device"],row["runtime"],row["boundary"],row["seed"]),[]).append(row)
        for i,(key,points) in enumerate(groups.items()):
            fig,ax=plt.subplots(figsize=(10,6))
            for row in points:
                ax.scatter(row["worst_repeat_p99_us"],row["equal_day_macro_f1"],marker="o" if row["numerical_status"]=="passed" else "x")
                ax.annotate(f"{row['model']} ({row['numerical_status']})",(row["worst_repeat_p99_us"],row["equal_day_macro_f1"]),fontsize=6)
            frontier=sorted([r for r in points if r["verified_frontier"]],key=lambda r:r["worst_repeat_p99_us"])
            if frontier: ax.plot([r["worst_repeat_p99_us"] for r in frontier],[r["equal_day_macro_f1"] for r in frontier])
            ax.set(xlabel="Worst-repeat p99 (µs)",ylabel="Equal-day validation macro-F1",title=f"{frozen['plan']['kind']} / {key[1]} / {key[2]} / {key[3]} / seed {key[4]}\n{key[0][:100]}")
            fig.tight_layout(); fig.savefig(output/f"accuracy-latency-{i}.png",dpi=150); plt.close(fig)
    return result
