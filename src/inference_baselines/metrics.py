import numpy as np


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def score(targets, predictions, probabilities=None):
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    matrix = np.bincount(3 * targets + predictions, minlength=9).reshape(3, 3)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=precision+recall != 0)
    result = {"accuracy": float(tp.sum()/len(targets)), "macro_f1": float(f1.mean()),
              "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist(),
              "support": support.tolist(), "confusion_matrix": matrix.tolist(),
              "zero_division": 0, "count": len(targets)}
    if probabilities is not None:
        p = probabilities[np.arange(len(targets)), targets]
        # A hard constant classifier has no declared probability convention.
        result["log_loss"] = float(-np.log(np.maximum(p, np.finfo(np.float64).tiny)).mean())
    return result


def compare(reference, candidate, atol, rtol):
    finite = np.isfinite(reference) & np.isfinite(candidate)
    both_rows = finite.all(axis=1)
    passed = (np.abs(np.subtract(candidate, reference, where=finite, out=np.full_like(reference, np.inf))) <= atol + rtol*np.abs(reference)).all(axis=1)
    errors = np.abs(reference[finite].astype(np.float64)-candidate[finite].astype(np.float64))
    relative = errors / np.maximum(np.abs(reference[finite]), np.finfo(np.float32).tiny)
    result = {"count": len(reference), "tolerance_pass_count": int((passed & both_rows).sum()),
              "finite_row_count": int(both_rows.sum()), "reference_nonfinite": int((~np.isfinite(reference)).sum()),
              "candidate_nonfinite": int((~np.isfinite(candidate)).sum()),
              "nonfinite_mask_disagreements": int((np.isfinite(reference) != np.isfinite(candidate)).sum()),
              "max_abs_error": float(errors.max()) if errors.size else None,
              "max_relative_error": float(relative.max()) if relative.size else None,
              "rms_error": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
              "atol": atol, "rtol": rtol,
              "class_disagreements_finite": int((reference[both_rows].argmax(1) != candidate[both_rows].argmax(1)).sum())}
    if both_rows.any():
        delta = np.abs(softmax(reference[both_rows])-softmax(candidate[both_rows]))
        result["probability_max_abs_error"] = float(delta.max())
        result["probability_rms_error"] = float(np.sqrt(np.mean(delta**2)))
    return result


def latency_summary(ns):
    ns = np.asarray(ns, dtype=np.float64)
    return {"count": len(ns), "p50_us": float(np.quantile(ns, .5)/1000),
            "p95_us": float(np.quantile(ns, .95)/1000), "p99_us": float(np.quantile(ns, .99)/1000),
            "maximum_us": float(ns.max()/1000), "service_calls_per_second": float(1e9/ns.mean()),
            "p999": "omitted: only ten observations beyond p99.9 at the default per-repeat budget"}
