from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox is outside valid geographic bounds")
    return west, south, east, north


def bbox_to_wkt(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    return (
        f"POLYGON(({west} {south},{east} {south},"
        f"{east} {north},{west} {north},{west} {south}))"
    )


def _property(product: Any, *names: str) -> Any:
    properties = getattr(product, "properties", None)
    for name in names:
        if isinstance(properties, dict) and name in properties:
            return properties[name]
        if hasattr(product, name):
            return getattr(product, name)
    return ""


def _product_row(product: Any) -> dict[str, str]:
    return {
        "scene_name": str(_property(product, "sceneName", "scene_name", "name")),
        "product_id": str(_property(product, "product_id", "productID", "granuleName")),
        "start_time": str(_property(product, "startTime", "start_time")),
        "stop_time": str(_property(product, "stopTime", "stop_time")),
        "beam_mode": str(_property(product, "beamMode", "beam_mode")),
        "polarization": str(_property(product, "polarization")),
        "processing_level": str(_property(product, "processingLevel", "processing_level")),
        "url": str(_property(product, "url", "downloadUrl", "download_url")),
    }


def search_products(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    max_results: int,
) -> Any:
    try:
        import asf_search as asf
    except ImportError as exc:
        raise RuntimeError(
            "Install the pipeline dependencies with "
            "python -m pip install -r data_pipeline/requirements.txt"
        ) from exc

    return asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1],
        intersectsWith=bbox_to_wkt(bbox),
        start=start,
        end=end,
        maxResults=max_results,
    )


def authenticate_if_requested() -> Any:
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    token = os.environ.get("EARTHDATA_TOKEN")
    if not (username and password) and not token:
        return None

    import asf_search as asf

    session = asf.ASFSession()
    if token:
        return session.auth_with_token(token)
    return session.auth_with_creds(username, password)


def write_results(path: Path, products: Any) -> list[dict[str, str]]:
    rows = [_product_row(product) for product in products]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "scene_name", "product_id", "start_time", "stop_time", "beam_mode",
            "polarization", "processing_level", "url",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Search and optionally download Sentinel-1 products from ASF.')
    parser.add_argument("--bbox", type=parse_bbox, required=True)
    parser.add_argument("--start", required=True, help="inclusive ISO date or datetime")
    parser.add_argument("--end", required=True, help="inclusive ISO date or datetime")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_results <= 0:
        raise SystemExit("--max-results must be positive")

    products = search_products(args.bbox, args.start, args.end, args.max_results)
    rows = write_results(args.out / "search_results.csv", products)
    print(f"Found {len(rows)} Sentinel-1 products")

    if args.download:
        session = authenticate_if_requested()
        if session is None:
            raise SystemExit(
                "Downloads require EARTHDATA_USERNAME/EARTHDATA_PASSWORD or "
                "EARTHDATA_TOKEN in the environment."
            )
        download_dir = args.out / "products"
        download_dir.mkdir(parents=True, exist_ok=True)
        products.download(path=str(download_dir), session=session, processes=1)
        print(f"Downloaded products to {download_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
