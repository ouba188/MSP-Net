"""Recompute exact metrics and repeated-run summaries from held-out predictions.

CSV columns: model, seed, sample_id, ship_group_id, port, fold,
true_label, pred_label. One prediction per sample/model/seed is required.
No run is generated or simulated by this tool.
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = {"model", "seed", "sample_id", "ship_group_id", "port", "fold", "true_label", "pred_label"}


def metrics(rows, classes):
    classes = tuple(classes)
    if not rows or len(set(classes)) != len(classes) or not classes:
        raise ValueError("Require observations and unique class labels")
    index = {name: i for i, name in enumerate(classes)}
    confusion = [[0] * len(classes) for _ in classes]
    for r in rows:
        if r["true_label"] not in index or r["pred_label"] not in index:
            raise ValueError("A prediction contains an undeclared class label")
        confusion[index[r["true_label"]]][index[r["pred_label"]]] += 1
    precision, recall, f1 = [], [], []
    for i in range(len(classes)):
        tp = confusion[i][i]
        actual, predicted = sum(confusion[i]), sum(row[i] for row in confusion)
        precision.append(tp / predicted if predicted else 0.0)
        recall.append(tp / actual if actual else 0.0)
        f1.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
    correct = sum(confusion[i][i] for i in range(len(classes)))
    return {"n": len(rows), "correct": correct, "accuracy": 100 * correct / len(rows),
            "macro_precision": 100 * statistics.mean(precision),
            "macro_recall": 100 * statistics.mean(recall),
            "macro_f1": 100 * statistics.mean(f1), "confusion": confusion,
            "class_support": {c: sum(confusion[i]) for i, c in enumerate(classes)}}


def summarize(values):
    return {"n_runs": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def evaluate(rows, classes, protocol="all-port"):
    runs = defaultdict(list)
    for r in rows:
        if REQUIRED - r.keys() or any(not str(r[k]).strip() for k in REQUIRED):
            raise ValueError("Missing required prediction fields")
        runs[r["model"], r["seed"]].append(r)
    if not runs:
        raise ValueError("Prediction file is empty")
    reference = None
    output = {"protocol": protocol, "classes": list(classes), "runs": [], "summary": {}}
    aggregated = defaultdict(lambda: defaultdict(list))
    for (model, seed), observations in sorted(runs.items()):
        ids, group_fold, signatures = set(), {}, {}
        for r in observations:
            if r["sample_id"] in ids:
                raise ValueError("Duplicate sample prediction within one model/seed")
            ids.add(r["sample_id"])
            group = r["ship_group_id"]
            # In LOPO a vessel present in two target ports would also occur in
            # source data unless explicitly purged; require externally purged splits.
            split_key = r["fold"] if protocol == "all-port" else r["port"]
            if protocol == "all-port" and group in group_fold and group_fold[group] != split_key:
                raise ValueError("Vessel identity appears in multiple outer held-out folds")
            group_fold[group] = split_key
            signatures[r["sample_id"]] = (r["true_label"], r["port"], group, r["fold"])
        if reference is None:
            reference = signatures
        elif signatures != reference:
            raise ValueError("Compared runs must use identical samples, labels and fixed splits")
        overall = metrics(observations, classes)
        by_port = defaultdict(list)
        for r in observations:
            by_port[r["port"]].append(r)
        port_scores = {p: metrics(v, classes) for p, v in sorted(by_port.items())}
        record = {"model": model, "seed": seed, "pooled": overall, "ports": port_scores}
        if protocol == "lopo":
            record["port_macro"] = {key: statistics.mean(s[key] for s in port_scores.values())
                                     for key in ("accuracy", "macro_f1")}
            selected = record["port_macro"]
        else:
            selected = overall
        for key in ("accuracy", "macro_f1"):
            aggregated[model][key].append(selected[key])
        output["runs"].append(record)
    output["summary"] = {model: {k: summarize(v) for k, v in scores.items()}
                         for model, scores in aggregated.items()}
    return output


def assert_identity_disjoint(train_groups, validation_groups, test_groups):
    partitions = [set(train_groups), set(validation_groups), set(test_groups)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Vessel identity leaks between train, validation and test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--protocol", choices=("all-port", "lopo"), default="all-port")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = evaluate(rows, args.classes, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluated {len(result['runs'])} real prediction sets; results: {args.output}")


if __name__ == "__main__":
    main()
