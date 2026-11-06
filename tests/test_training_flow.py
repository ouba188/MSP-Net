import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mspnet.priors import ScalePriorEncoder
from mspnet.training import (PreparedData, TrainingConfig, fit_preprocessing,
                             fit_size, outer_splits, run_protocol, _batches)


def prepared(n=75):
    rng = np.random.default_rng(3)
    return PreparedData(
        sar=rng.random((n, 3, 2, 2), dtype=np.float32),
        lonlat=np.zeros((n, 2), dtype=np.float32),
        local_position=rng.normal(size=(n, 3)).astype(np.float32),
        osm=rng.normal(size=(n, 2)).astype(np.float32),
        osm_availability=np.ones((n, 1), dtype=np.float32),
        osm_availability_index=np.array([0, 0]),
        osm_feature_names=np.array(["log_distance", "density"]),
        osm_availability_names=np.array(["available"]),
        labels=np.arange(n) % 3, class_names=np.array(["a", "b", "c"]),
        size_targets=np.tile(np.array([60., 10.], dtype=np.float32), (n, 1)),
        sample_ids=np.array([f"sample_{i}" for i in range(n)]),
        group_ids=np.array([f"group_{i}" for i in range(n)]),
        ports=np.array([f"port_{i % 3}" for i in range(n)]),
        preprocessing_json=json.dumps({"purpose": "synthetic behavior test only"}),
    )


class TinySize(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 4)

    def forward(self, sar):
        output = self.layer(sar[:, :, 0, 0])
        return torch.cat((F.softplus(output[:, :2]), output[:, 2:].clamp(-12, 12)), 1)


class TinyClassifier(nn.Module):
    def __init__(self, osm_dim, num_classes, osm_availability_dim=0):
        super().__init__()
        self.scale_encoder = ScalePriorEncoder(token_dim=4, hidden_dim=4)
        self.linear = nn.Linear(3 + 4 + osm_dim, num_classes)
        self.projection = nn.Linear(3, 4)

    def forward(self, sar, size_stats, lonlat, local_position, osm_features,
                osm_availability, return_alignment=None):
        size = self.scale_encoder(size_stats.detach())
        visual = sar[:, :, 0, 0]
        return {"logits": self.linear(torch.cat((visual, size, osm_features), 1)),
                "visual_alignment": F.normalize(self.projection(visual), dim=-1) if return_alignment else None,
                "prior_alignment": F.normalize(size, dim=-1) if return_alignment else None}


class TinyVisual(nn.Module):
    def __init__(self, osm_dim, num_classes, osm_availability_dim=0, fusion_kind="visual"):
        super().__init__()
        self.linear = nn.Linear(3, num_classes)

    def forward(self, sar, size_stats, lonlat, local_position, osm_features,
                osm_availability, return_alignment=None):
        return {"logits": self.linear(sar[:, :, 0, 0]),
                "visual_alignment": None, "prior_alignment": None}


class TrainingFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_lopo_purges_source_identity_without_removing_target(self):
        data = prepared()
        data.group_ids[1] = data.group_ids[0]
        split, source, target = next(x for x in outer_splits(data, TrainingConfig(protocol="lopo")) if x[0] == "port_0")
        self.assertIn(0, target)
        self.assertNotIn(1, source)
        self.assertFalse(set(data.group_ids[source]) & set(data.group_ids[target]))

    def test_statistics_use_only_update_and_missing_remains_zero(self):
        data = prepared(3)
        data.local_position[:] = [[1, 2, 3], [3, 4, 5], [10000, 10000, 10000]]
        data.osm[:] = [[2, 4], [0, 0], [9999, 9999]]
        data.osm_availability[1] = 0
        _, osm, stats = fit_preprocessing(data, np.array([0, 1]))
        self.assertEqual(stats["local_mean"], [2, 3, 4])
        self.assertEqual(stats["osm_mean"], [2, 4])
        np.testing.assert_array_equal(osm[1], [0, 0])

    def test_contrastive_tail_is_merged_without_dropping_samples(self):
        batches = list(_batches(np.arange(17), 8, merge_single=True))
        self.assertEqual([len(batch) for batch in batches], [8, 9])
        np.testing.assert_array_equal(np.concatenate(batches), np.arange(17))

    def test_training_seeds_do_not_change_fixed_outer_splits(self):
        data = prepared()
        left = outer_splits(data, TrainingConfig(seed=17))
        right = outer_splits(data, TrainingConfig(seed=42))
        for (_, train_a, test_a), (_, train_b, test_b) in zip(left, right):
            np.testing.assert_array_equal(train_a, train_b)
            np.testing.assert_array_equal(test_a, test_b)

    def test_visual_baseline_never_fits_or_reports_size_estimation(self):
        data = prepared()
        config = TrainingConfig(size_epochs=1, epochs=1, patience=1,
                                fusion_options={"fusion_kind": "visual"})
        with tempfile.TemporaryDirectory() as folder:
            with patch("mspnet.training.fit_size", side_effect=AssertionError("Unexpected size fitting")):
                records = run_protocol(data, config, folder, TinyVisual, TinySize)
            output = Path(folder)
            self.assertEqual(len(records), len(data.labels))
            self.assertFalse((output / "size_predictions.csv").exists())
            self.assertFalse((output / "size_errors.json").exists())
            self.assertFalse((output / "split_0" / "size_estimator.pt").exists())
            self.assertEqual(json.loads((output / "config.json").read_text())["lambda_v"], 0)
            self.assertFalse(json.loads((output / "split_0" / "split.json").read_text())["size_stage_used"])

    def test_allport_training_records_no_heldout_size_supervision(self):
        data = prepared()
        config = TrainingConfig(size_epochs=1, epochs=1, patience=1)
        size_fit_calls = []

        def traced(data, indices, config, factory, seed):
            size_fit_calls.append(set(indices.tolist()))
            return fit_size(data, indices, config, factory, seed)

        with tempfile.TemporaryDirectory() as folder, patch("mspnet.training.fit_size", side_effect=traced):
            records = run_protocol(data, config, folder, TinyClassifier, TinySize)
            output = Path(folder)
            self.assertEqual(len(records), len(data.labels))
            self.assertEqual(len({r["sample_id"] for r in records}), len(data.labels))
            splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            for fold, split in enumerate(splits):
                update = set(split["update"])
                forbidden = set(split["validation"]) | set(split["test"])
                calls = size_fit_calls[fold * 6:(fold + 1) * 6]
                self.assertEqual(calls[-1], update)
                self.assertTrue(all(call <= update and not call & forbidden for call in calls))
                held = [i for inner in split["size_cross_fitting"] for i in inner["held_out"]]
                self.assertEqual(set(held), update)
                self.assertEqual(len(held), len(update))
                for inner in split["size_cross_fitting"]:
                    self.assertFalse(set(inner["fit"]) & set(inner["held_out"]))
                checkpoint = torch.load(output / f"split_{fold}" / "classifier.pt", weights_only=False)
                self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(len(size_fit_calls), 30)
            size_errors = json.loads((output / "size_errors.json").read_text())["by_view"]
            self.assertEqual(size_errors["outer_test"]["pooled"]["n"], len(data.labels))
            self.assertEqual(set(size_errors["outer_test"]["pooled_by_class"]), {"a", "b", "c"})
            self.assertGreater(size_errors["source_validation"]["pooled"]["n"], 0)
            self.assertIn("by_class", size_errors["source_validation"]["per_split"]["0"])
            self.assertTrue((output / "identity_splits.json").exists())
            self.assertTrue((output / "predictions.csv").exists())
            from tools.evaluate_predictions import evaluate
            identity = json.loads((output / "identity_splits.json").read_text(encoding="utf-8"))
            report = evaluate(records, data.class_names.tolist(), identity_splits=identity, expected_folds=5)
            self.assertTrue(report["runs"][0]["training_identity_verified"])
            self.assertIn("fold_summary", report["runs"][0])


if __name__ == "__main__":
    unittest.main()
