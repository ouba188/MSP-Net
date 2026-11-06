import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mspnet.training import PreparedData
from tools.prepare_training import prepare


class PrepareTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, root):
        Image.fromarray(np.full((64, 64), 1, dtype=np.uint8)).save(root / "vv.png")
        Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(root / "vh.png")
        row = {"sample_id": "one", "port_id": "port", "label": "tanker", "ship_group_id": "group_one",
               "lon": "120", "lat": "30", "length_m": "60", "width_m": "10",
               "chip_path_vv": "vv.png", "chip_path_vh": "vh.png"}
        with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "pier1",
                   "properties": {"man_made": "pier"}, "geometry": {"type": "Point", "coordinates": [120, 30]}}]}
        (root / "port.geojson").write_text(json.dumps(geojson), encoding="utf-8")
        (root / "ports.json").write_text(json.dumps({"port": {"reference_lonlat": [120, 30], "geojson": "port.geojson"}}), encoding="utf-8")

    def test_explicit_divisor_channels_and_prepared_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            for divisor in (1, 255):
                output = root / f"prepared_{divisor}.npz"
                prepare(root / "manifest.csv", root, root / "ports.json", ["bulk", "tanker"], divisor, output)
                data = PreparedData.from_npz(output)
                data.validate()
                self.assertEqual(data.sar.shape, (1, 3, 224, 224))
                expected = np.array([np.log1p(1 / divisor), np.log1p(255 / divisor),
                                     np.log1p(255 / divisor) - np.log1p(1 / divisor)])
                np.testing.assert_allclose(data.sar[0, :, 100, 100], expected, rtol=1e-6)
                self.assertEqual(data.labels.tolist(), [1])
                self.assertEqual(data.sample_ids.tolist(), ["one"])
                self.assertEqual(data.osm.shape, (1, 68))
                self.assertEqual(data.osm_availability.tolist(), [[0, 1, 0, 0, 0]])
                np.testing.assert_allclose(data.local_position, 0, atol=1e-6)
                metadata = json.loads(data.preprocessing_json)
                self.assertEqual(metadata["intensity_divisor"], divisor)
                self.assertEqual(metadata["ports"]["port"]["reference_lonlat"], [120, 30])

    def test_rejects_unrecorded_scale_and_wrong_chip_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "explicit PNG intensity divisor"):
                prepare(root / "manifest.csv", root, root / "ports.json", ["tanker"], 256, root / "bad.npz")
            Image.fromarray(np.zeros((32, 64), dtype=np.uint8)).save(root / "vv.png")
            with self.assertRaisesRegex(ValueError, "single-channel 64x64"):
                prepare(root / "manifest.csv", root, root / "ports.json", ["tanker"], 255, root / "bad.npz")


if __name__ == "__main__":
    unittest.main()
