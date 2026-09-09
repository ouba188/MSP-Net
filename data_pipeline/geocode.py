"""Rate-limited, cached geocoding for a small place-name CSV."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import requests


RESULT_FIELDS = ("latitude", "longitude", "display_name", "osm_type", "osm_id", "geocode_error")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_cache(path: Path, query_column: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_rows(path)
    return {
        str(row.get(query_column, "")).strip(): row
        for row in rows
        if str(row.get(query_column, "")).strip()
        and str(row.get("latitude", "")).strip()
        and str(row.get("longitude", "")).strip()
    }


def geocode_one(
    session: requests.Session,
    endpoint: str,
    query: str,
    timeout: float,
) -> dict[str, str]:
    response = session.get(
        endpoint,
        params={"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1},
        timeout=timeout,
    )
    response.raise_for_status()
    results: list[dict[str, Any]] = response.json()
    if not results:
        return {field: "" for field in RESULT_FIELDS} | {"geocode_error": "no result"}
    result = results[0]
    return {
        "latitude": str(result.get("lat", "")),
        "longitude": str(result.get("lon", "")),
        "display_name": str(result.get("display_name", "")),
        "osm_type": str(result.get("osm_type", "")),
        "osm_id": str(result.get("osm_id", "")),
        "geocode_error": "",
    }


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-column", default="query")
    parser.add_argument(
        "--endpoint",
        default="https://nominatim.openstreetmap.org/search",
    )
    parser.add_argument("--user-agent", default="MSPNet-public-research/1.0 (+https://github.com/ouba188/MSP-Net)")
    parser.add_argument("--min-interval", type=float, default=1.1)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_interval < 1.0:
        raise SystemExit("--min-interval must be at least 1 second for the public service")
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit("input CSV is empty")
    if args.query_column not in rows[0]:
        raise SystemExit(f"input CSV has no {args.query_column!r} column")

    fields = list(rows[0])
    for field in RESULT_FIELDS:
        if field not in fields:
            fields.append(field)
    cache = load_cache(args.output, args.query_column)
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent, "Accept-Language": "en"})
    last_request = 0.0

    for index, row in enumerate(rows, start=1):
        query = str(row.get(args.query_column, "")).strip()
        if not query:
            row["geocode_error"] = "empty query"
        elif query in cache:
            row.update({field: cache[query].get(field, "") for field in RESULT_FIELDS})
        else:
            wait = args.min_interval - (time.monotonic() - last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                row.update(geocode_one(session, args.endpoint, query, args.timeout))
            except requests.RequestException as exc:
                row.update({field: "" for field in RESULT_FIELDS})
                row["geocode_error"] = f"{type(exc).__name__}: {exc}"
            last_request = time.monotonic()
            cache[query] = dict(row)
        write_rows(args.output, rows, fields)
        print(f"Processed {index}/{len(rows)}: {query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

