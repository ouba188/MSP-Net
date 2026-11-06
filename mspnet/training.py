"""Executable, identity-disjoint two-stage training for the manuscript.

Prepared NPZ contract (no raw AIS or unprocessed SAR is inferred here):
  sar: float32 [N,3,H,W], already log-compressed/contrast-constructed/resized;
  lonlat: degrees [N,2]; local_position: unstandardized port-relative UTM [N,3];
  osm: [N,D] continuous features, distance log1p/density/coverage already applied;
  osm_availability: [N,A] binary; osm_availability_index: [D], column-to-flag
    mapping (-1 means always available). Missing continuous values are zero.
  osm_feature_names / osm_availability_names: fixed ordered Unicode columns;
  labels: integer class indices; class_names: ordered Unicode class names;
  size_targets: AIS length/width in metres [N,2]; sample_ids/group_ids/ports: [N];
  preprocessing_json: scalar JSON string recording the actual SAR/geographic
    preparation, including intensity scale and port reference coordinates.

Classifier monitoring groups are separated BEFORE size fitting. Only update
groups supply size supervision or fitted statistics. No target labels select
epochs. Defaults for stage one are a runnable configuration, not recovered
historical experiment records. The command never claims to reproduce a score.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from .losses import heteroscedastic_size_loss, mspnet_loss
from .model import MSPNet
from .size_estimator import SARSizeEstimator


@dataclass
class TrainingConfig:
    protocol: str = "all-port"
    seed: int = 42
    split_seed: int = 42
    folds: int = 5
    size_epochs: int = 100
    size_learning_rate: float = 1e-4
    size_weight_decay: float = 1e-4
    size_batch_size: int = 32
    epochs: int = 220
    patience: int = 130
    learning_rate: float = 2e-5
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    batch_size: int = 8
    label_smoothing: float = 0.08
    lambda_v: float = 5e-4
    contrastive_temperature: float = 0.5
    gradient_norm: float = 1.0
    model_name: str = "MSP-Net"
    device: str = "cpu"
    fusion_options: dict = field(default_factory=dict)


@dataclass
class PreparedData:
    sar: np.ndarray
    lonlat: np.ndarray
    local_position: np.ndarray
    osm: np.ndarray
    osm_availability: np.ndarray
    osm_availability_index: np.ndarray
    osm_feature_names: np.ndarray
    osm_availability_names: np.ndarray
    labels: np.ndarray
    class_names: np.ndarray
    size_targets: np.ndarray
    sample_ids: np.ndarray
    group_ids: np.ndarray
    ports: np.ndarray
    preprocessing_json: str

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: str(data[name].item()) if name == "preprocessing_json"
                          else data[name] for name in cls.__dataclass_fields__})

    def validate(self):
        n = len(self.labels)
        if self.sar.ndim != 4 or self.sar.shape[:2] != (n, 3):
            raise ValueError("sar must be [N,3,H,W] with prepared channels")
        for name, width in (("lonlat", 2), ("local_position", 3), ("size_targets", 2)):
            if getattr(self, name).shape != (n, width):
                raise ValueError(f"{name} must have shape [N,{width}]")
        for name in ("sar", "lonlat", "local_position", "osm", "size_targets"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"Non-finite prepared {name}")
        if (self.size_targets <= 0).any():
            raise ValueError("AIS length/width supervision must be positive")
        if self.osm.ndim != 2 or self.osm.shape[0] != n:
            raise ValueError("osm must be [N,D]")
        if self.osm_availability.ndim != 2 or self.osm_availability.shape[0] != n:
            raise ValueError("osm_availability must be [N,A]")
        d, a = self.osm.shape[1], self.osm_availability.shape[1]
        if len(self.osm_feature_names) != d or len(self.osm_availability_names) != a:
            raise ValueError("OSM column names must describe the complete fixed schema")
        if self.osm_availability_index.shape != (d,):
            raise ValueError("Provide one availability index per continuous OSM column")
        if not np.isin(self.osm_availability, [0, 1]).all():
            raise ValueError("Availability channels must be binary")
        if ((self.osm_availability_index < -1) | (self.osm_availability_index >= a)).any():
            raise ValueError("Invalid OSM availability mapping")
        if not np.issubdtype(self.labels.dtype, np.integer) or not np.isin(self.labels, range(len(self.class_names))).all():
            raise ValueError("labels must index the fixed class_names array")
        for name in ("sample_ids", "group_ids", "ports"):
            values = getattr(self, name)
            if values.shape != (n,) or any(not str(v).strip() for v in values):
                raise ValueError(f"{name} must contain N nonempty identifiers")
        if len(set(self.sample_ids)) != n:
            raise ValueError("sample_ids must be unique")
        if not isinstance(json.loads(self.preprocessing_json), dict):
            raise ValueError("preprocessing_json must describe preparation as a JSON object")


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grouped_splits(data, indices, folds, seed):
    indices = np.asarray(indices, dtype=int)
    if len(set(data.group_ids[indices])) < folds:
        raise ValueError(f"Need at least {folds} independent vessel groups")
    splitter = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    result = []
    for train, held in splitter.split(indices, data.labels[indices], data.group_ids[indices]):
        if not len(train) or not len(held):
            raise ValueError("Empty grouped partition; dataset cannot support this protocol")
        result.append((indices[train], indices[held]))
    return result


def outer_splits(data, config):
    indices = np.arange(len(data.labels))
    if config.protocol == "all-port":
        return [(str(fold), train, test) for fold, (train, test) in enumerate(
            grouped_splits(data, indices, config.folds, config.split_seed))]
    if config.protocol != "lopo":
        raise ValueError("protocol must be all-port or lopo")
    result = []
    for port in sorted(set(data.ports)):
        test = indices[data.ports == port]
        target_groups = set(data.group_ids[test])
        train = np.array([i for i in indices if data.ports[i] != port and data.group_ids[i] not in target_groups])
        result.append((str(port), train, test))
    return result


def _tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _batches(indices, batch_size, shuffle=False, merge_single=False):
    order = np.random.permutation(indices) if shuffle else np.asarray(indices)
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    if merge_single and len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = np.concatenate((batches[-2], batches[-1]))
        batches.pop()
    yield from batches


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_size(data, indices, config, factory, seed):
    _seed(seed)
    model = factory().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.size_learning_rate, weight_decay=config.size_weight_decay)
    model.train()
    for _ in range(config.size_epochs):
        for ids in _batches(indices, config.size_batch_size, shuffle=True):
            target = _tensor(data.size_targets[ids], config.device)
            loss = heteroscedastic_size_loss(model(_tensor(data.sar[ids], config.device)), target[:, 0], target[:, 1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_norm)
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict_size(model, data, indices, config):
    return torch.cat([model(_tensor(data.sar[ids], config.device)).cpu()
                      for ids in _batches(indices, config.size_batch_size)], 0).numpy()


def fit_preprocessing(data, update):
    local_mean = data.local_position[update].mean(0)
    local_std = np.maximum(data.local_position[update].std(0), 1e-6)
    mean, std = np.zeros(data.osm.shape[1]), np.ones(data.osm.shape[1])
    availability = data.osm_availability_index.astype(int)
    for col, flag in enumerate(availability):
        selected = update if flag < 0 else update[data.osm_availability[update, flag] == 1]
        if len(selected):
            mean[col] = data.osm[selected, col].mean()
            std[col] = max(float(data.osm[selected, col].std()), 1e-6)
    local = (data.local_position - local_mean) / local_std
    osm = (data.osm - mean) / std
    for col, flag in enumerate(availability):
        if flag >= 0:
            osm[data.osm_availability[:, flag] == 0, col] = 0
    return local, osm, {"local_mean": local_mean.tolist(), "local_std": local_std.tolist(),
                        "osm_mean": mean.tolist(), "osm_std": std.tolist()}


def _forward(model, data, ids, size_stats, local, osm, device, alignment):
    return model(_tensor(data.sar[ids], device), _tensor(size_stats[ids], device),
                 _tensor(data.lonlat[ids], device), _tensor(local[ids], device),
                 _tensor(osm[ids], device), _tensor(data.osm_availability[ids], device),
                 return_alignment=alignment)


def train_classifier(data, update, validation, size_stats, config, factory, directory):
    _seed(config.seed)
    model = factory(osm_dim=data.osm.shape[1], num_classes=len(data.class_names),
                    osm_availability_dim=data.osm_availability.shape[1], **config.fusion_options).to(config.device)
    local, osm, statistics = fit_preprocessing(data, update)
    if getattr(model, "scale_encoder", None) is not None:
        model.scale_encoder.set_descriptor_statistics(_tensor(size_stats[update], config.device))
        statistics.update(scale_mean=model.scale_encoder.descriptor_mean.tolist(),
                          scale_std=model.scale_encoder.descriptor_std.tolist())
    _write_json(directory / "statistics.json", statistics)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs, eta_min=config.minimum_learning_rate)
    best, stale, history = float("inf"), 0, []
    for epoch in range(config.epochs):
        model.train()
        train_loss, count = 0.0, 0
        for ids in _batches(update, config.batch_size, shuffle=True, merge_single=True):
            output = _forward(model, data, ids, size_stats, local, osm, config.device, True)
            loss, _ = mspnet_loss(output["logits"], torch.as_tensor(data.labels[ids], dtype=torch.long, device=config.device),
                                 output["visual_alignment"], output["prior_alignment"], config.lambda_v,
                                 config.contrastive_temperature, config.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_norm)
            optimizer.step()
            train_loss += loss.item() * len(ids)
            count += len(ids)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for ids in _batches(validation, config.batch_size):
                output = _forward(model, data, ids, size_stats, local, osm, config.device, False)
                # Classification CE selects checkpoints; contrastive batches do
                # not define a stable per-sample validation objective.
                loss = F.cross_entropy(output["logits"], torch.as_tensor(data.labels[ids], dtype=torch.long, device=config.device),
                                       label_smoothing=config.label_smoothing, reduction="sum")
                validation_loss += loss.item()
        validation_loss /= len(validation)
        history.append({"epoch": epoch + 1, "training_loss": train_loss / count,
                        "validation_ce": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "validation_ce": best,
                        "config": asdict(config), "statistics": statistics,
                        "osm_feature_names": data.osm_feature_names.tolist(),
                        "osm_availability_names": data.osm_availability_names.tolist(),
                        "osm_availability_index": data.osm_availability_index.tolist()}, directory / "classifier.pt")
        else:
            stale += 1
        schedule.step()
        if stale >= config.patience:
            break
    checkpoint = torch.load(directory / "classifier.pt", map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _write_json(directory / "training_history.json", history)
    return model, local, osm


def run_protocol(data, config, output_dir, classifier_factory=MSPNet, size_factory=SARSizeEstimator):
    data.validate()
    uses_size = config.fusion_options.get("fusion_kind", "fpcsa") != "visual"
    if config.fusion_options.get("fusion_kind", "fpcsa") != "fpcsa":
        names = {"visual": "Visual baseline", "concat": "Direct concatenation", "film": "FiLM", "vanilla": "Vanilla cross-attention"}
        name = names.get(config.fusion_options["fusion_kind"], config.model_name) if config.model_name == "MSP-Net" else config.model_name
        config = replace(config, lambda_v=0.0, model_name=name)
    if config.folds < 2 or min(config.size_epochs, config.epochs, config.patience) < 1 or config.batch_size < 2:
        raise ValueError("Require multiple folds, positive epochs/patience and classification batch_size >= 2")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", {**asdict(config), "stage1_default_provenance": "new executable defaults; not historical run evidence",
                                        "preprocessing": json.loads(data.preprocessing_json)})
    records, size_records, identity_records, split_records = [], [], [], []
    for split, outer_train, test in outer_splits(data, config):
        directory = output / f"split_{len(split_records)}"
        directory.mkdir(exist_ok=True)
        update, validation = grouped_splits(data, outer_train, config.folds, config.split_seed)[0]
        if len(update) < 2:
            raise ValueError("Classifier updates need at least two samples")
        partition = {name: ids.tolist() for name, ids in (("update", update), ("validation", validation), ("test", test))}
        partition.update(split=split, sample_ids={name: data.sample_ids[ids].tolist() for name, ids in
                                                (("update", update), ("validation", validation), ("test", test))})
        # All three sets must be separated before ANY supervised size fitting.
        group_sets = [set(data.group_ids[ids]) for ids in (update, validation, test)]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Identity overlap between update/validation/test")
        identity_records.append({"model": config.model_name, "seed": str(config.seed), "split": split,
                                 **{key: sorted(groups) for key, groups in zip(
                                     ("train_groups", "validation_groups", "test_groups"), group_sets)}})
        stats = np.zeros((len(data.labels), 4), dtype=np.float32)
        inner_records = []
        if uses_size:
            for inner, (fit_ids, held_ids) in enumerate(grouped_splits(data, update, config.folds, config.split_seed)):
                size = fit_size(data, fit_ids, config, size_factory, config.seed + inner)
                stats[held_ids] = predict_size(size, data, held_ids, config)
                inner_records.append({"fit": fit_ids.tolist(), "held_out": held_ids.tolist()})
                del size
            size = fit_size(data, update, config, size_factory, config.seed)
            stats[validation] = predict_size(size, data, validation, config)
            stats[test] = predict_size(size, data, test, config)
            torch.save({"model": size.state_dict(), "fit_sample_ids": data.sample_ids[update].tolist(),
                        "config": asdict(config)}, directory / "size_estimator.pt")
            del size
            np.savez_compressed(directory / "size_predictions.npz", sample_ids=data.sample_ids[np.concatenate((update, validation, test))],
                                size_stats=stats[np.concatenate((update, validation, test))])
        if uses_size:
            for view, indices in (("source_validation", validation), ("outer_test", test)):
                for i in indices:
                    size_records.append({"sample_id": str(data.sample_ids[i]), "split": split, "view": view,
                                         "class_label": str(data.class_names[data.labels[i]]), "port": str(data.ports[i]),
                                         "true_length_m": float(data.size_targets[i, 0]), "true_width_m": float(data.size_targets[i, 1]),
                                         "pred_length_m": float(np.expm1(stats[i, 0])), "pred_width_m": float(np.expm1(stats[i, 1]))})
        partition["size_cross_fitting"] = inner_records
        partition["size_stage_used"] = uses_size
        split_records.append(partition)
        _write_json(directory / "split.json", partition)
        model, local, osm = train_classifier(data, update, validation, stats, config, classifier_factory, directory)
        with torch.no_grad():
            for ids in _batches(test, config.batch_size):
                pred = _forward(model, data, ids, stats, local, osm, config.device, False)["logits"].argmax(-1).cpu().numpy()
                for i, predicted in zip(ids, pred):
                    records.append({"model": config.model_name, "seed": str(config.seed), "sample_id": str(data.sample_ids[i]),
                                    "ship_group_id": str(data.group_ids[i]), "port": str(data.ports[i]), "fold": split,
                                    "true_label": str(data.class_names[data.labels[i]]), "pred_label": str(data.class_names[predicted])})
        del model
        _write_json(output / "splits.json", split_records)
        _write_json(output / "identity_splits.json", identity_records)
        for filename, rows in (("predictions.csv", records), ("size_predictions.csv", size_records)):
            if not rows:
                continue
            with (output / filename).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if not uses_size:
        return records
    _write_json(output / "size_errors.json", {
        "dimension_order": ["length", "width"],
        "summary_axis": "fold" if config.protocol == "all-port" else "port",
        "weighting": {"pooled": "equal prediction-observation weight, no deduplication across splits",
                      "summary": "equal split weight"},
        "source_validation_note": "Validation observations may repeat across outer runs; retain split identifiers. These are not unique outer OOF samples.",
        "by_view": {view: summarize_size_errors([row for row in size_records if row["view"] == view])
                    for view in ("source_validation", "outer_test")},
    })
    return records


def summarize_size_errors(rows):
    """Compute metre errors per prediction observation, separately by view."""
    errors = np.array([[r["pred_length_m"] - r["true_length_m"],
                       r["pred_width_m"] - r["true_width_m"]] for r in rows])

    def score(values):
        return {"n": len(values), "mae_m": np.abs(values).mean(0).tolist(),
                "rmse_m": np.sqrt((values ** 2).mean(0)).tolist()}

    per_split = {}
    for split in sorted({r["split"] for r in rows}):
        selected = np.array([r["split"] == split for r in rows])
        per_split[split] = score(errors[selected])
        per_split[split]["by_class"] = {
            label: score(errors[selected & np.array([r["class_label"] == label for r in rows])])
            for label in sorted({r["class_label"] for r in rows if r["split"] == split})}
    pooled_by_class = {label: score(errors[[r["class_label"] == label for r in rows]])
                       for label in sorted({r["class_label"] for r in rows})}
    summary = {}
    for metric in ("mae_m", "rmse_m"):
        values = np.array([scores[metric] for scores in per_split.values()])
        summary[metric] = {"mean": values.mean(0).tolist(),
                           "sample_std": values.std(0, ddof=1).tolist() if len(values) > 1 else None}
    return {"per_split": per_split, "summary": summary,
            "pooled": score(errors), "pooled_by_class": pooled_by_class}
