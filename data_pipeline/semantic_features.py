"""Fixed, prospective OSM feature schema for the revised manuscript.

GeoJSON must contain WGS84 geometries and OSM tags in feature properties
(either flat or under ``tags``). Use a geometry-aware converter for Overpass
relations: ``python -m data_pipeline.semantic_features convert-overpass
--input overpass.json --output port.geojson`` requires osm2geojson.

Extraction keeps 68 continuous columns and 5 availability flags separate.
Only distance is log1p-transformed here. Fit all standardization on training
samples in mspnet.training. This schema is not a claim about archived runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely import transform, union_all

GROUPS = ("shoreline_protection", "wharf_berth", "waterway_anchorage",
          "port_industrial", "oil_facilities")
RADII_M = (200, 1000, 5000)
NEIGHBORHOOD_METRICS = ("count", "density_per_m2", "area_ratio", "line_density_per_m")
COLUMNS = tuple(name for group in GROUPS for name in
                (f"{group}.nearest_log1p_m",) + tuple(
                    f"{group}.{radius}m.{metric}" for radius in RADII_M
                    for metric in NEIGHBORHOOD_METRICS)) + (
    "valid_feature_ratio", "missing_feature_ratio", "map_polygon_coverage")
AVAILABILITY_INDEX = np.array([i for i in range(len(GROUPS)) for _ in range(13)] + [-1] * 3,
                              dtype=np.int64)


def semantic_groups(tags):
    """An object may belong to multiple groups; groups are not class labels."""
    result = []
    if tags.get("natural") == "coastline" or tags.get("man_made") == "breakwater":
        result.append(0)
    if (tags.get("man_made") in ("pier", "quay") or tags.get("seamark:type") == "berth"
            or tags.get("waterway") == "dock"):
        result.append(1)
    if (tags.get("seamark:type") in ("fairway", "anchorage")
            or tags.get("waterway") in ("river", "canal", "fairway")):
        result.append(2)
    if (tags.get("landuse") == "industrial" or tags.get("industrial") == "port"
            or tags.get("harbour") == "terminal"):
        result.append(3)
    if (tags.get("man_made") in ("storage_tank", "pipeline") or tags.get("pipeline")
            or tags.get("substance") == "oil" or tags.get("industrial") == "oil"):
        result.append(4)
    return result


def port_transformer(reference_lonlat):
    """Choose one UTM CRS from the fixed port reference, never per vessel."""
    lon, lat = map(float, reference_lonlat)
    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError("Port reference must be a valid WGS84 position within UTM latitude bounds")
    zone = min(60, int((lon + 180) // 6) + 1)
    epsg = (32600 if lat >= 0 else 32700) + zone
    return Transformer.from_crs(4326, epsg, always_xy=True), epsg


def geometry_parts(geometry, kinds):
    if geometry.geom_type in kinds:
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for component in geometry.geoms for part in geometry_parts(component, kinds)]
    return []


def extract_semantic_features(geojson, lonlat, reference_lonlat):
    """Return raw local coordinates, OSM columns, flags, and column mapping.

    For each vessel, a semantic group is available iff at least one of its
    mapped objects intersects the 5 km neighborhood. Missing groups have 13
    zeros and flag 0; no imputed physical distance is inferred. Nearest
    distances are to geometry (not centroid), restricted to that neighborhood.
    Counts count feature objects once, including non-point geometries.
    Areas/line lengths are unioned to avoid overlapping geometry double counts.
    Polygon boundaries do not contribute to line density. Overlay unions use
    a 1 mm precision grid to remove projection roundoff at shared segments.
    """
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Expected a standard WGS84 GeoJSON FeatureCollection")
    lonlat = np.asarray(lonlat, dtype=np.float64)
    if lonlat.ndim != 2 or lonlat.shape[1] != 2 or not np.isfinite(lonlat).all():
        raise ValueError("lonlat must contain finite [longitude, latitude] rows")
    transformer, epsg = port_transformer(reference_lonlat)
    ref_x, ref_y = transformer.transform(*reference_lonlat)
    x, y = transformer.transform(lonlat[:, 0].tolist(), lonlat[:, 1].tolist())
    x, y = np.asarray(x), np.asarray(y)
    local_xy = np.column_stack((x - ref_x, y - ref_y))
    local_position = np.column_stack((local_xy, np.linalg.norm(local_xy, axis=1)))
    grouped = [[] for _ in GROUPS]
    all_polygons = []
    seen_ids = set()
    for feature in geojson.get("features", []):
        properties = feature.get("properties") or {}
        # Standard converters preserve feature ids; repeated identical OSM
        # objects from merged exports must not inflate object counts.
        feature_id = feature.get("id")
        if feature_id is None and "type" in properties and "id" in properties:
            feature_id = (properties["type"], properties["id"])
        if feature_id is not None:
            if feature_id in seen_ids:
                continue
            seen_ids.add(feature_id)
        if feature.get("geometry") is None:
            raise ValueError("GeoJSON contains an object without geometry; export full geometries")
        geometry = transform(shape(feature["geometry"]), lambda xy: np.column_stack(
            transformer.transform(xy[:, 0].tolist(), xy[:, 1].tolist())))
        if geometry.is_empty:
            continue
        if not geometry.is_valid:
            raise ValueError(f"Invalid mapped geometry for feature {feature_id}; repair the source geometry")
        tags = properties.get("tags", properties)
        for index in semantic_groups(tags):
            grouped[index].append(geometry)
        all_polygons.extend(geometry_parts(geometry, ("Polygon",)))
    mapped_polygons = union_all(all_polygons, grid_size=0.001)
    osm = np.zeros((len(lonlat), len(COLUMNS)), dtype=np.float64)
    availability = np.zeros((len(lonlat), len(GROUPS)), dtype=np.float64)
    for sample, (east, north) in enumerate(zip(x, y)):
        point = Point(east, north)
        neighborhoods = [point.buffer(radius, quad_segs=64) for radius in RADII_M]
        for group_index, geometries in enumerate(grouped):
            nearby = [g for g in geometries if g.distance(point) <= RADII_M[-1]]
            if not nearby:
                continue
            availability[sample, group_index] = 1
            base = 13 * group_index
            osm[sample, base] = math.log1p(min(g.distance(point) for g in nearby))
            polygons = union_all([p for g in nearby for p in geometry_parts(g, ("Polygon",))], grid_size=0.001)
            lines = union_all([p for g in nearby for p in geometry_parts(g, ("LineString", "LinearRing"))], grid_size=0.001)
            for radius_index, (radius, window) in enumerate(zip(RADII_M, neighborhoods)):
                area = math.pi * radius ** 2
                count = sum(g.intersects(window) for g in nearby)
                values = (count, count / area, polygons.intersection(window).area / area,
                          lines.intersection(window).length / area)
                start = base + 1 + radius_index * 4
                osm[sample, start:start + 4] = values
        valid = availability[sample].mean()
        osm[sample, -3:] = (valid, 1 - valid,
                           mapped_polygons.intersection(neighborhoods[-1]).area / (math.pi * RADII_M[-1] ** 2))
    return {"local_position": local_position.astype(np.float32), "osm": osm.astype(np.float32),
            "osm_availability": availability.astype(np.float32),
            "osm_availability_index": AVAILABILITY_INDEX.copy(),
            "osm_columns": np.asarray(COLUMNS), "availability_columns": np.asarray(GROUPS),
            "reference_lonlat": np.asarray(reference_lonlat, dtype=np.float64),
            "utm_epsg": np.asarray(epsg), "radii_m": np.asarray(RADII_M)}


def convert_overpass(payload):
    """Delegate way/relation reconstruction to the explicit optional dependency."""
    try:
        from osm2geojson import json2geojson
    except ImportError as exc:
        raise ImportError("Overpass conversion requires: python -m pip install osm2geojson") from exc
    result = json2geojson(payload)
    if result.get("type") != "FeatureCollection":
        raise ValueError("Converter did not produce a GeoJSON FeatureCollection")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-overpass")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--geojson", type=Path, required=True)
    extract.add_argument("--samples", type=Path, required=True,
                         help="CSV with sample_id, longitude, latitude, containing one port")
    extract.add_argument("--reference-lon", type=float, required=True)
    extract.add_argument("--reference-lat", type=float, required=True)
    extract.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "convert-overpass":
        result = convert_overpass(json.loads(args.input.read_text(encoding="utf-8")))
        args.output.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        with args.samples.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        lonlat = [[float(row["longitude"]), float(row["latitude"])] for row in rows]
        result = extract_semantic_features(json.loads(args.geojson.read_text(encoding="utf-8")),
                                           lonlat, (args.reference_lon, args.reference_lat))
        result["sample_id"] = np.asarray([row["sample_id"] for row in rows])
        np.savez_compressed(args.output, **result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
