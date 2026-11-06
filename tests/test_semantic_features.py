import importlib.util
import math
import unittest

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point, box, mapping
from shapely import transform

from data_pipeline.osm_overpass import make_query
from data_pipeline.semantic_features import (COLUMNS, extract_semantic_features,
                                              port_transformer, semantic_groups, convert_overpass)


class SemanticFeatureTests(unittest.TestCase):
    reference = (123., 30.)

    def setUp(self):
        forward, self.epsg = port_transformer(self.reference)
        self.x0, self.y0 = forward.transform(*self.reference)
        self.backward = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def lonlat(self, x, y):
        return self.backward.transform(self.x0 + x, self.y0 + y)

    def feature(self, geometry, tags, identity):
        def to_wgs84(xy):
            return np.column_stack(self.backward.transform((xy[:, 0] + self.x0).tolist(),
                                                           (xy[:, 1] + self.y0).tolist()))
        return {"type": "Feature", "id": identity, "properties": tags,
                "geometry": mapping(transform(geometry, to_wgs84))}

    def extract(self, features, points=None):
        return extract_semantic_features({"type": "FeatureCollection", "features": features},
                                         [self.reference] if points is None else points, self.reference)

    def column(self, result, name):
        return result["osm"][0, COLUMNS.index(name)]

    def test_fixed_schema_empty_map_and_shared_metric_position(self):
        result = self.extract([], [self.reference, self.lonlat(300, 400)])
        self.assertEqual(result["osm"].shape, (2, 68))
        self.assertEqual(result["osm_availability"].shape, (2, 5))
        np.testing.assert_array_equal(result["osm_availability"], 0)
        np.testing.assert_array_equal(result["osm"][:, :65], 0)
        np.testing.assert_array_equal(result["osm"][:, -3:], [[0, 1, 0], [0, 1, 0]])
        np.testing.assert_allclose(result["local_position"], [[0, 0, 0], [300, 400, 500]], atol=1e-4)
        np.testing.assert_array_equal(result["osm_availability_index"], np.repeat(np.arange(5), 13).tolist() + [-1]*3)
        self.assertEqual(int(result["utm_epsg"]), 32651)

    def test_distance_object_counts_densities_and_five_km_availability(self):
        features = [self.feature(Point(distance, 0), {"substance": "oil"}, f"node/{i}")
                    for i, distance in enumerate((100, 400, 6000))]
        result = self.extract(features)
        self.assertAlmostEqual(self.column(result, "oil_facilities.nearest_log1p_m"), math.log1p(100), places=5)
        self.assertEqual(result["osm_availability"][0].tolist(), [0, 0, 0, 0, 1])
        for radius, expected_count in ((200, 1), (1000, 2), (5000, 2)):
            self.assertEqual(self.column(result, f"oil_facilities.{radius}m.count"), expected_count)
            self.assertAlmostEqual(self.column(result, f"oil_facilities.{radius}m.density_per_m2"),
                                   expected_count / (math.pi * radius**2), delta=1e-10)
        far_only = self.extract(features[-1:])
        self.assertEqual(far_only["osm_availability"].sum(), 0)
        self.assertEqual(far_only["osm"][0, :65].sum(), 0)

    def test_polygon_union_avoids_overlap_and_boundaries_are_not_lines(self):
        features = [self.feature(box(0, 0, 100, 100), {"landuse": "industrial"}, "way/1"),
                    self.feature(box(50, 0, 150, 100), {"industrial": "port"}, "way/2")]
        # A duplicated export id is still one mapped object.
        result = self.extract(features + [features[0]])
        self.assertEqual(self.column(result, "port_industrial.200m.count"), 2)
        self.assertAlmostEqual(self.column(result, "port_industrial.200m.area_ratio"),
                               15000 / (math.pi * 200**2), places=6)
        self.assertEqual(self.column(result, "port_industrial.200m.line_density_per_m"), 0)
        self.assertAlmostEqual(self.column(result, "map_polygon_coverage"),
                               15000 / (math.pi * 5000**2), delta=1e-10)

    def test_line_density_unions_overlaps_without_polygon_boundaries(self):
        features = [self.feature(LineString([(0, 0), (100, 0)]), {"natural": "coastline"}, "way/1"),
                    self.feature(LineString([(50, 0), (150, 0)]), {"man_made": "breakwater"}, "way/2")]
        result = self.extract(features)
        self.assertEqual(self.column(result, "shoreline_protection.200m.count"), 2)
        self.assertAlmostEqual(self.column(result, "shoreline_protection.200m.line_density_per_m"),
                               150 / (math.pi * 200**2), delta=1e-9)
        self.assertEqual(self.column(result, "shoreline_protection.200m.area_ratio"), 0)

    def test_tags_cover_each_manuscript_group_and_overpass_query(self):
        examples = [({"natural": "coastline"}, 0), ({"waterway": "dock"}, 1),
                    ({"seamark:type": "anchorage"}, 2), ({"landuse": "industrial"}, 3),
                    ({"industrial": "port"}, 3), ({"harbour": "terminal"}, 3),
                    ({"pipeline": "oil"}, 4), ({"substance": "oil"}, 4),
                    ({"industrial": "oil"}, 4), ({"man_made": "storage_tank"}, 4)]
        for tags, group in examples:
            self.assertIn(group, semantic_groups(tags))
        query = make_query((120, 29, 124, 32), 120)
        for tag in ('"landuse"="industrial"', '"industrial"', '"pipeline"',
                    '"substance"="oil"', 'dock', '"seamark:type"'):
            self.assertIn(tag, query)

    @unittest.skipUnless(importlib.util.find_spec("osm2geojson"), "optional Overpass converter not installed")
    def test_optional_overpass_converter_preserves_tagged_geometry(self):
        payload = {"version": 0.6, "elements": [{"type": "node", "id": 1,
                    "lat": 30., "lon": 123., "tags": {"man_made": "storage_tank"}}]}
        result = extract_semantic_features(convert_overpass(payload), [self.reference], self.reference)
        self.assertEqual(result["osm_availability"][0, 4], 1)
        self.assertEqual(self.column(result, "oil_facilities.200m.count"), 1)


if __name__ == "__main__":
    unittest.main()
