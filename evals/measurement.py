"""Pure accounting helpers for calibration; no model calls or inferred human labels."""
from collections import Counter
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
import re


LABELS = {"supported", "unsupported", "contradicted"}


def wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - margin), min(1, center + margin)]


def rate(successes: int, total: int) -> dict:
    return {"numerator": successes, "denominator": total,
            "value": successes / total if total else None, "wilson95": wilson(successes, total)}


def metrics(rows: list[dict], include_failures: bool) -> dict:
    selected = rows if include_failures else [row for row in rows if row["actual"] is not None]
    expected = [row["expected"] for row in selected]
    actual = [row["actual"] for row in selected]
    tp = sum(e == "unsupported" and a == "unsupported" for e, a in zip(expected, actual))
    fp = sum(e != "unsupported" and a == "unsupported" for e, a in zip(expected, actual))
    fn = sum(e == "unsupported" and a != "unsupported" for e, a in zip(expected, actual))
    # Failure on a non-positive claim is unclassified, not a true negative.
    tn = sum(e != "unsupported" and a is not None and a != "unsupported" for e, a in zip(expected, actual))
    return {
        "claims": len(selected), "class_counts": dict(Counter(expected)),
        "prediction_counts": dict(Counter(a if a is not None else "failed" for a in actual)),
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "failed_nonpositive": sum(e != "unsupported" and a is None for e, a in zip(expected, actual)),
        "precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn),
        "agreement": rate(sum(e == a for e, a in zip(expected, actual)), len(selected)),
    }


def load_labels(dataset_path: Path, labels_path: Path) -> tuple[dict, dict, dict]:
    raw = dataset_path.read_bytes()
    label_raw = labels_path.read_bytes()
    dataset = json.loads(raw)
    form = json.loads(label_raw)
    if form["dataset_sha256"] != sha256(raw).hexdigest():
        raise ValueError("Human labels refer to a different dataset")
    required = [case["id"] + "/" + claim["id"] for case in dataset["cases"] for claim in case["claims"]]
    if len(required) != len(set(required)):
        raise ValueError("Dataset has duplicate claim identifiers")
    labels = {row["id"]: row["label"] for row in form["claims"]}
    if set(labels) != set(required) or len(labels) != len(form["claims"]):
        raise ValueError("Human labels must cover each claim exactly once")
    if any(label not in LABELS for label in labels.values()):
        raise ValueError("Finish all human labels before calibration")
    return dataset, labels, {
        "dataset_sha256": sha256(raw).hexdigest(), "labels_sha256": sha256(label_raw).hexdigest(),
        "labels_mtime_ns": labels_path.stat().st_mtime_ns,
    }


def claim_covered(seed: str, extracted: str) -> bool:
    """Casefolded word-token span containment, or >=90% seed-token multiset recall."""
    wanted = re.findall(r"\w+", seed.casefold())
    found = re.findall(r"\w+", extracted.casefold())
    if not wanted:
        return False
    if (" " + " ".join(wanted) + " ") in (" " + " ".join(found) + " "):
        return True
    overlap = sum((Counter(wanted) & Counter(found)).values())
    return overlap / len(wanted) >= 0.9


def summarize(rows: list[dict]) -> dict:
    groups = {"combined": rows}
    for stratum in ("seeded", "natural"):
        groups[stratum] = [row for row in rows if row["stratum"] == stratum]
    result = {}
    for name, group in groups.items():
        supported = [row for row in group if row["expected"] == "supported"]
        unsafe = [row for row in group if row["expected"] in {"unsupported", "contradicted"}]
        result[name] = {
            "excluding_failures": metrics(group, False),
            "failures_as_misses": metrics(group, True),
            "specificity": {
                "supported_false_alarm": rate(sum(row["flagged"] for row in supported), len(supported)),
                "unsupported_or_contradicted_hit": rate(sum(row["flagged"] for row in unsafe), len(unsafe)),
            },
        }
    return result
