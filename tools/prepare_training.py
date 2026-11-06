"""Prepare training arrays from the companion manifest and mapped port objects.

The manifest is produced by data_pipeline.build_manifest. Port configuration
is a JSON object keyed by port_id; each value contains reference_lonlat
([longitude, latitude]) and geojson (path relative to that JSON file, or
absolute). References must be fixed independently of experiment splits.

Example:
  python -m tools.prepare_training --manifest manifest.csv --dataset-root chips
    --ports ports.json --classes bulk_carrier container_ship tanker
    --intensity-divisor 255 --output outputs/prepared.npz

The divisor is REQUIRED: PNG sample scaling is not inferred from image maxima
and is not a claim of radiometric calibration. Output contains local sample
identifiers and must remain outside version control. The default outputs/
directory is already ignored by the repository.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data_pipeline.semantic_features import (COLUMNS, GROUPS, RADII_M,
                                             AVAILABILITY_INDEX, extract_semantic_features)
from mspnet.sar_input import build_three_channel
from mspnet.training import PreparedData


REQUIRED = {"sample_id", "port_id", "label", "ship_group_id", "lon", "lat",
            "length_m", "width_m", "chip_path_vv", "chip_path_vh"}


def prepare(manifest, dataset_root, ports_config, class_names, intensity_divisor, output):
    """Write a PreparedData NPZ; continuous-feature standardization is deferred.

    The returned dictionary is preparation metadata, not experiment results.
    Input images must be paired, single-channel 64x64 PNG chips. SAR channels
    are constructed at 64x64, then resized together to 224x224.
    """
    if intensity_divisor not in (1, 255):
        raise ValueError("Choose an explicit PNG intensity divisor of 1 or 255")
    class_names = tuple(class_names)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Provide a unique ordered list of class names")
    manifest, dataset_root, ports_config, output = map(Path, (manifest, dataset_root, ports_config, output))
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no samples")
    for line, row in enumerate(rows, 2):
        if any(not str(row.get(key) or "").strip() for key in REQUIRED):
            raise ValueError(f"Manifest line {line} has empty required fields")
        if row["label"] not in class_names:
            raise ValueError(f"Unknown class {row['label']!r} on manifest line {line}")
    ids = np.asarray([row["sample_id"] for row in rows])
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest sample_id must be unique")
    config = json.loads(ports_config.read_text(encoding="utf-8"))
    port_ids = np.asarray([row["port_id"] for row in rows])
    missing_ports = set(port_ids) - config.keys()
    if missing_ports:
        raise ValueError(f"Missing fixed port references/GeoJSON for: {sorted(missing_ports)}")
    lonlat = np.array([[float(row["lon"]), float(row["lat"])] for row in rows], dtype=np.float64)
    dimensions = np.array([[float(row["length_m"]), float(row["width_m"])] for row in rows], dtype=np.float32)
    if not np.isfinite(lonlat).all() or (np.abs(lonlat[:, 0]) > 180).any() or (np.abs(lonlat[:, 1]) > 90).any():
        raise ValueError("Manifest longitude/latitude must be finite WGS84 degrees")
    if not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("Manifest length_m/width_m must be positive finite metre values")

    n = len(rows)
    sar = np.empty((n, 3, 224, 224), dtype=np.float32)
    for i, row in enumerate(rows):
        channels = []
        for key in ("chip_path_vv", "chip_path_vh"):
            path = dataset_root / row[key]
            with Image.open(path) as image:
                array = np.array(image, dtype=np.float32, copy=True)
            if array.shape != (64, 64):
                raise ValueError(f"{row['sample_id']} {key}: expected single-channel 64x64 chip, got {array.shape}")
            if not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f"Invalid PNG sample values in {path}")
            channels.append(torch.from_numpy(array / float(intensity_divisor)))
        sar[i] = build_three_channel(channels[0], channels[1], output_size=224)[0].numpy()

    local = np.empty((n, 3), dtype=np.float32)
    osm = np.empty((n, len(COLUMNS)), dtype=np.float32)
    flags = np.empty((n, len(GROUPS)), dtype=np.float32)
    port_metadata = {}
    for port in sorted(set(port_ids)):
        details = config[port]
        reference = details["reference_lonlat"]
        geojson_path = ports_config.parent / details["geojson"]
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        selected = np.flatnonzero(port_ids == port)
        features = extract_semantic_features(geojson, lonlat[selected], reference)
        local[selected] = features["local_position"]
        osm[selected] = features["osm"]
        flags[selected] = features["osm_availability"]
        port_metadata[str(port)] = {"reference_lonlat": list(map(float, reference)),
                                    "utm_epsg": int(features["utm_epsg"]),
                                    "geojson_path": str(geojson_path.resolve())}
    metadata = {
        "manifest_path": str(manifest.resolve()), "dataset_root": str(dataset_root.resolve()),
        "samples": n, "input_storage": "paired single-channel 64x64 PNG",
        "intensity_divisor": intensity_divisor,
        "intensity_interpretation": "PNG sample values divided by explicit divisor; no radiometric calibration inferred",
        "sar_order": ["divide PNG values", "log1p each polarization", "VH_log minus VV_log", "bilinear resize to 224x224"],
        "sar_channels": ["log1p(VV)", "log1p(VH)", "log1p(VH)-log1p(VV)"],
        "resize_align_corners": False, "distance_transform": "log1p(metres)",
        "neighborhood_radii_m": list(RADII_M), "ports": port_metadata,
        "standardization": "not fitted here; fit only classifier-update samples in each nested run",
        "schema_origin": "new explicit public implementation; not evidence of historical experiment settings",
    }
    prepared = PreparedData(
        sar=sar, lonlat=lonlat.astype(np.float32), local_position=local, osm=osm,
        osm_availability=flags, osm_availability_index=AVAILABILITY_INDEX.copy(),
        osm_feature_names=np.asarray(COLUMNS), osm_availability_names=np.asarray(GROUPS),
        labels=np.array([class_names.index(row["label"]) for row in rows], dtype=np.int64),
        class_names=np.asarray(class_names), size_targets=dimensions, sample_ids=ids,
        group_ids=np.asarray([row["ship_group_id"] for row in rows]), ports=port_ids,
        preprocessing_json=json.dumps(metadata, ensure_ascii=False),
    )
    prepared.validate()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid asdict(): it would deep-copy the full SAR array before writing.
    np.savez(output, **{name: getattr(prepared, name) for name in PreparedData.__dataclass_fields__})
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--ports", type=Path, required=True, help="JSON mapping port_id to reference_lonlat and geojson")
    parser.add_argument("--classes", nargs="+", required=True, help="Explicit fixed class order")
    parser.add_argument("--intensity-divisor", type=int, choices=(1, 255), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/prepared.npz"))
    args = parser.parse_args()
    metadata = prepare(args.manifest, args.dataset_root, args.ports, args.classes, args.intensity_divisor, args.output)
    print(f"Prepared {metadata['samples']} samples: {args.output}")


if __name__ == "__main__":
    main()
