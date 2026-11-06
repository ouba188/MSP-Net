from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


BASE_FIELDS = (
    "sample_id",
    "dataset_name",
    "port_id",
    "port_name",
    "sar_product_id",
    "chip_path_vv",
    "chip_path_vh",
    "has_vv",
    "has_vh",
    "polar_mode_available",
    "label",
    "ship_group_id",
    "lon",
    "lat",
    "length_m",
    "width_m",
    "length_width_ratio",
    "quality_flag",
)
PRIVATE_COLUMNS = {
    "mmsi",
    "mmsi_or_ship_id",
    "imo",
    "imo_number",
    "timestamp",
    "track",
    "track_json",
    "ais_raw",
}
ALLOWED_EXTRA_PREFIXES = ("osm_", "dist_to_", "rel_angle_", "in_berth_", "coverage_", "availability_")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build and validate a relative-path manifest for prepared dual-pol chips.')
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", default="SAR-AIS")
    parser.add_argument("--stored-size", type=int, default=64)
    parser.add_argument(
        "--classes",
        default="bulk_carrier,container_ship,tanker",
        help="comma-separated class order used for validation",
    )
    parser.add_argument("--summary-out", type=Path)
    return parser.parse_args()


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, object]]:
    input_rows = load_rows(args.metadata)
    if not input_rows:
        raise ValueError("metadata CSV is empty")
    input_fields = set(input_rows[0])
    forbidden = sorted({field for field in input_fields if field.lower() in PRIVATE_COLUMNS})
    if forbidden:
        raise ValueError(f"raw identity or track columns are not accepted: {forbidden}")
    required = {"sample_id", "port_id", "label", "ship_group_id"}
    missing = sorted(required - input_fields)
    if missing:
        raise ValueError(f"metadata CSV is missing required columns: {missing}")

    allowed_classes = {item.strip() for item in args.classes.split(",") if item.strip()}
    if not allowed_classes:
        raise ValueError("--classes must contain at least one class")
    output_rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    missing_images: list[str] = []
    bad_sizes: list[str] = []

    for source in input_rows:
        sample_id = str(source["sample_id"]).strip()
        port_id = str(source["port_id"]).strip()
        label = str(source["label"]).strip()
        group_id = str(source["ship_group_id"]).strip()
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"sample_id is empty or duplicated: {sample_id!r}")
        if label not in allowed_classes:
            raise ValueError(f"unknown label {label!r} for sample {sample_id}")
        if not group_id:
            raise ValueError(f"ship_group_id is required for sample {sample_id}")
        seen_ids.add(sample_id)

        vv = args.dataset_root / "data" / port_id / "vv" / f"{sample_id}.png"
        vh = args.dataset_root / "data" / port_id / "vh" / f"{sample_id}.png"
        if not vv.exists() or not vh.exists():
            missing_images.append(sample_id)
            continue
        if args.stored_size > 0:
            if image_size(vv) != (args.stored_size, args.stored_size) or image_size(vh) != (args.stored_size, args.stored_size):
                bad_sizes.append(sample_id)
                continue

        relative_vv = vv.relative_to(args.dataset_root).as_posix()
        relative_vh = vh.relative_to(args.dataset_root).as_posix()
        row = {
            "sample_id": sample_id,
            "dataset_name": str(source.get("dataset_name") or args.dataset_name),
            "port_id": port_id,
            "port_name": str(source.get("port_name") or port_id),
            "sar_product_id": str(source.get("sar_product_id", "")),
            "chip_path_vv": relative_vv,
            "chip_path_vh": relative_vh,
            "has_vv": "1",
            "has_vh": "1",
            "polar_mode_available": "vv,vh",
            "label": label,
            "ship_group_id": group_id,
        }
        for field in ("lon", "lat", "length_m", "width_m", "quality_flag"):
            if field in source:
                row[field] = str(source[field])
        if "length_width_ratio" in source and source["length_width_ratio"]:
            row["length_width_ratio"] = str(source["length_width_ratio"])
        elif row.get("length_m") and row.get("width_m"):
            try:
                row["length_width_ratio"] = str(float(row["length_m"]) / float(row["width_m"]))
            except ValueError:
                row["length_width_ratio"] = ""
        for field in input_fields:
            if field.startswith(ALLOWED_EXTRA_PREFIXES):
                row[field] = str(source.get(field, ""))
        output_rows.append(row)

    if missing_images or bad_sizes:
        raise ValueError(
            f"chip validation failed: missing_pairs={len(missing_images)}, "
            f"wrong_size={len(bad_sizes)}"
        )

    extra_fields = sorted({field for row in output_rows for field in row if field not in BASE_FIELDS})
    fields = list(BASE_FIELDS) + extra_fields
    output_rows.sort(key=lambda row: (row["port_id"], row["sample_id"]))
    summary = {
        "rows": len(output_rows),
        "ports": dict(sorted(Counter(row["port_id"] for row in output_rows).items())),
        "labels": dict(sorted(Counter(row["label"] for row in output_rows).items())),
        "stored_size": args.stored_size,
        "numeric_derived_fields": len([field for field in extra_fields if field.startswith(("osm_", "dist_to_"))]),
        "relative_paths_only": True,
    }
    write_rows(args.output, output_rows, fields)
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_rows, summary


def main() -> int:
    args = parse_args()
    if args.stored_size < 0:
        raise SystemExit("--stored-size must be non-negative")
    _, summary = build_manifest(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
