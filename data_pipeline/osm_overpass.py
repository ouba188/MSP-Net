"""Fetch a bounded set of OSM semantic objects for a port area."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    try:
        west, south, east, north = (float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north") from exc
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise argparse.ArgumentTypeError("bbox is outside valid geographic bounds")
    return west, south, east, north


def make_query(bbox: tuple[float, float, float, float], timeout: int) -> str:
    west, south, east, north = bbox
    box = f"{south},{west},{north},{east}"
    return f"""[out:json][timeout:{timeout}];
(
  nwr[\"man_made\"~\"quay|pier|breakwater|crane|silo|storage_tank|works\"]({box});
  nwr[\"harbour\"]({box});
  nwr[\"seamark:type\"]({box});
  nwr[\"waterway\"~\"^(river|canal|fairway|dock)$\"]({box});
  nwr[\"landuse\"=\"industrial\"]({box});
  nwr[\"industrial\"~\"^(port|oil)$\"]({box});
  nwr[\"pipeline\"]({box});
  nwr[\"substance\"=\"oil\"]({box});
  nwr[\"man_made\"=\"pipeline\"]({box});
  way[\"natural\"=\"coastline\"]({box});
);
out body geom;"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox", type=parse_bbox, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://overpass-api.de/api/interpreter")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--user-agent", default="MSPNet-public-research/1.0 (+https://github.com/ouba188/MSP-Net)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0 or args.request_timeout <= 0:
        raise SystemExit("timeouts must be positive")
    query = make_query(args.bbox, args.timeout)
    response = requests.post(
        args.endpoint,
        data={"data": query},
        headers={"User-Agent": args.user_agent},
        timeout=args.request_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    output = {
        "source": "OpenStreetMap contributors via Overpass API",
        "endpoint": args.endpoint,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "bbox_wsen": list(args.bbox),
        "query": query,
        "elements": payload.get("elements", []),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(output['elements'])} OSM elements to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

