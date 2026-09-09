# MSP-Net Data Pipeline Helpers

This directory contains small, public-facing utilities for acquiring source
data and assembling a manifest. They are deliberately separate from the core
model implementation.

## Included utilities

- `sentinel1_search_download.py`: search the Alaska Satellite Facility catalog
  for Sentinel-1 products and optionally download them;
- `geocode.py`: geocode a small list of port names or other non-sensitive
  place queries with a cached, rate-limited Nominatim client;
- `osm_overpass.py`: retrieve tagged OSM objects for a specified port bounding
  box from Overpass;
- `build_manifest.py`: validate paired VV/VH chips and create a relative-path
  manifest suitable for the released dataset layout.

Install the optional dependencies with:

```bash
python -m pip install -r data_pipeline/requirements.txt
```

## Typical commands

Search Sentinel-1 products over a small bounding box:

```bash
python -m data_pipeline.sentinel1_search_download \
  --bbox "129.5,35.0,129.9,35.5" \
  --start 2025-01-01 --end 2025-01-31 \
  --out data_pipeline/outputs/busan_s1
```

Add `--download` to download the selected products. ASF downloads normally
require Earthdata authentication. Provide credentials through
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or configure the authentication
method supported by `asf_search`; no credentials are read from source files.

Geocode a small CSV containing a `query` column:

```bash
python -m data_pipeline.geocode \
  --input ports.csv --output data_pipeline/outputs/ports_geocoded.csv
```

The public Nominatim service is rate-limited and must not be used for bulk or
systematic queries. The utility enforces a one-request-per-second minimum,
identifies itself with a User-Agent, caches completed rows, and should only be
used for a small, non-sensitive place list. See the
[Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/).

Retrieve OSM objects around a port:

```bash
python -m data_pipeline.osm_overpass \
  --bbox "129.5,35.0,129.9,35.5" \
  --out data_pipeline/outputs/busan_osm.json
```

The query is intentionally limited to infrastructure, harbor, waterway,
coastline, and seamark tags relevant to the OSM semantic prior. The output is
raw Overpass JSON and can be converted to the numeric feature table used by a
downstream preparation script. Attribute OSM data to OpenStreetMap
contributors and follow the [ODbL terms](https://www.openstreetmap.org/copyright).

Build a public manifest from a metadata CSV and a dataset root containing
`data/<port>/vv/<sample_id>.png` and `data/<port>/vh/<sample_id>.png`:

```bash
python -m data_pipeline.build_manifest \
  --metadata labels.csv \
  --dataset-root /path/to/SAR-AIS-Dataset \
  --output /path/to/SAR-AIS-Dataset/metadata/manifest.csv
```

The metadata CSV must contain `sample_id`, `port_id`, `label`, and
`ship_group_id`. Optional columns such as coordinates, physical size,
`quality_flag`, and numeric fields prefixed by `osm_` or `dist_to_` are copied
as derived metadata. Raw MMSI/IMO values, tracks, timestamps, and absolute
file paths are rejected. The builder does not publish raw AIS or SAR source
records; it only validates already prepared chips and writes relative paths.

## Scope and provenance

These scripts are reusable acquisition and preparation helpers, not a dump of
the private cloud training workspace. They do not include raw AIS records,
private annotations, internal launchers, checkpoints, or experiment logs.
The released SAR-AIS repository contains the derived seven-port chips and
metadata used with the paper:

<https://github.com/ouba188/SAR-AIS-Dataset>

