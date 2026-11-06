"""Report fold/port mean and sample SD from held-out predictions.

CSV columns: model, seed, sample_id, ship_group_id, port, fold,
true_label, pred_label. One prediction per sample/model/seed is required.
All-port scores average held-out folds. LOPO reports each port across seeds,
then averages those port means across ports. Per-run port summaries and the
separate seed summary retain their own axes. Pooled metrics are diagnostic.
Optional identity-splits JSON is a list of records with model, seed, split,
train_groups, validation_groups, test_groups. Split is fold (all-port) or port
(LOPO). No run is generated or simulated by this tool.
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

REQUIRED = {"model", "seed", "sample_id", "ship_group_id", "port", "fold", "true_label", "pred_label"}
SCORE_KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


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
    absent = []
    for i in range(len(classes)):
        tp = confusion[i][i]
        actual, predicted = sum(confusion[i]), sum(row[i] for row in confusion)
        if actual == 0 and predicted == 0:
            absent.append(classes[i])
        # The manuscript fixes the macro denominator to the declared classes.
        # Undefined class precision/recall/F1 use zero, including absent classes.
        precision.append(tp / predicted if predicted else 0.0)
        recall.append(tp / actual if actual else 0.0)
        f1.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
    if not precision:
        raise ValueError("No declared class has support or predictions")
    correct = sum(confusion[i][i] for i in range(len(classes)))
    return {"n": len(rows), "correct": correct, "accuracy": 100 * correct / len(rows),
            "macro_precision": 100 * statistics.mean(precision),
            "macro_recall": 100 * statistics.mean(recall),
            "macro_f1": 100 * statistics.mean(f1), "confusion": confusion,
            "macro_class_count": len(precision), "absent_classes": absent,
            "class_support": {c: sum(confusion[i]) for i, c in enumerate(classes)}}


def summarize(values, unit="runs"):
    return {f"n_{unit}": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def summarize_scores(scores, unit):
    return {key: summarize([score[key] for score in scores.values()], unit)
            for key in SCORE_KEYS}


def evaluate(rows, classes, protocol="all-port", identity_splits=None, expected_folds=None):
    if protocol not in ("all-port", "lopo"):
        raise ValueError("Protocol must be all-port or lopo")
    classes = tuple(classes)
    splits = {}
    if identity_splits is not None:
        for split in identity_splits:
            key = tuple(str(split[k]) for k in ("model", "seed", "split"))
            if key in splits:
                raise ValueError("Duplicate identity split record")
            assert_identity_disjoint(split["train_groups"], split["validation_groups"], split["test_groups"])
            splits[key] = split
    runs = defaultdict(list)
    for r in rows:
        if REQUIRED - r.keys() or any(not str(r[k]).strip() for k in REQUIRED):
            raise ValueError("Missing required prediction fields")
        runs[r["model"], r["seed"]].append(r)
    if not runs:
        raise ValueError("Prediction file is empty")
    reference = None
    output = {"protocol": protocol, "classes": list(classes), "runs": [], "seed_summary": {}}
    aggregated = defaultdict(lambda: defaultdict(list))
    for (model, seed), observations in sorted(runs.items()):
        ids, group_fold, group_ports, signatures = set(), {}, {}, {}
        by_fold, by_port = defaultdict(list), defaultdict(list)
        for r in observations:
            if r["sample_id"] in ids:
                raise ValueError("Duplicate sample prediction within one model/seed")
            ids.add(r["sample_id"])
            group = r["ship_group_id"]
            if protocol == "all-port" and group in group_fold and group_fold[group] != r["fold"]:
                raise ValueError("Vessel identity appears in multiple outer held-out folds")
            group_fold[group] = r["fold"]
            group_ports.setdefault(group, set()).add(r["port"])
            signatures[r["sample_id"]] = (r["true_label"], r["port"], group, r["fold"])
            by_fold[r["fold"]].append(r)
            by_port[r["port"]].append(r)
        if reference is None:
            reference = signatures
        elif signatures != reference:
            raise ValueError("Compared runs must use identical samples, labels and fixed splits")
        if protocol == "all-port" and expected_folds is not None and len(by_fold) != expected_folds:
            raise ValueError(f"Expected {expected_folds} held-out folds, found {len(by_fold)}")
        test_splits = by_fold if protocol == "all-port" else by_port
        if identity_splits is not None:
            for name, held_out in test_splits.items():
                key = (str(model), str(seed), str(name))
                if key not in splits:
                    raise ValueError(f"Missing train/validation/test identity record for {key}")
                reported = set(map(str, splits[key]["test_groups"]))
                actual = {str(r["ship_group_id"]) for r in held_out}
                if reported != actual:
                    raise ValueError(f"Test identity record does not match predictions for {key}")
        port_scores = {p: metrics(v, classes) for p, v in sorted(by_port.items())}
        record = {"model": model, "seed": seed, "pooled": metrics(observations, classes),
                  "ports": port_scores, "training_identity_verified": identity_splits is not None}
        if protocol == "lopo":
            overlapping = {g: sorted(p) for g, p in group_ports.items() if len(p) > 1}
            if overlapping:
                # Prediction CSVs cannot establish train/test leakage. Source
                # training can exclude each target vessel even across ports.
                record["identity_overlap"] = {"groups": len(overlapping),
                    "examples": dict(list(overlapping.items())[:5]),
                    "meaning": "cross-target-port overlap; not a train/test leakage verdict"}
            record["port_summary"] = summarize_scores(port_scores, "ports")
            selected = record["port_summary"]
        else:
            fold_scores = {f: metrics(v, classes) for f, v in sorted(by_fold.items())}
            record["folds"] = fold_scores
            record["fold_summary"] = summarize_scores(fold_scores, "folds")
            selected = record["fold_summary"]
        for key in SCORE_KEYS:
            aggregated[model][key].append(selected[key]["mean"])
        output["runs"].append(record)
    output["seed_summary"] = {model: {k: summarize(v, "seeds") for k, v in scores.items()}
                              for model, scores in aggregated.items()}
    if protocol == "lopo":
        # The manuscript reports each target port over repeated seeds, then
        # computes the aggregate row over those port-level means. Its SD is
        # across port means, not across seeds or all port/seed observations.
        by_model_port = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for record in output["runs"]:
            for port, score in record["ports"].items():
                for key in SCORE_KEYS:
                    by_model_port[record["model"]][port][key].append(score[key])
        output["lopo_summary"] = {}
        for model, ports in by_model_port.items():
            port_repeats = {port: {key: summarize(values, "seeds")
                                   for key, values in scores.items()}
                            for port, scores in ports.items()}
            port_means = {port: {key: summary["mean"] for key, summary in scores.items()}
                          for port, scores in port_repeats.items()}
            output["lopo_summary"][model] = {
                "ports": port_repeats,
                "port_mean_summary": summarize_scores(port_means, "ports"),
            }
    return output


def assert_identity_disjoint(train_groups, validation_groups, test_groups):
    partitions = [set(map(str, groups)) for groups in (train_groups, validation_groups, test_groups)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Vessel identity leaks between train, validation and test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--protocol", choices=("all-port", "lopo"), default="all-port")
    parser.add_argument("--expected-folds", type=int, default=5,
                        help="required number of outer held-out folds for all-port (default: 5)")
    parser.add_argument("--identity-splits", type=Path,
                        help="JSON train/validation/test vessel identities for each evaluation split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.predictions.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    identity_splits = json.loads(args.identity_splits.read_text(encoding="utf-8")) if args.identity_splits else None
    result = evaluate(rows, args.classes, args.protocol, identity_splits, args.expected_folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluated {len(result['runs'])} real prediction sets; results: {args.output}")


if __name__ == "__main__":
    main()
