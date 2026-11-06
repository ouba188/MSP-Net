import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluation_protocol", Path(__file__).parents[1] / "tools/evaluate_predictions.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def rows_for_units(units):
    rows = []
    for name, correct, count in units:
        for _ in range(count):
            idx = str(len(rows))
            rows.append(dict(model="m", seed="42", sample_id=idx,
                             ship_group_id="g" + idx, port=name, fold=name,
                             true_label="a", pred_label="a" if correct else "b"))
    return rows


class EvaluationProtocolTests(unittest.TestCase):
    def test_five_fold_mean_is_not_pooled_accuracy(self):
        # Scores [100,0,100,0,100] with unequal fold sizes. Hand calculation:
        # fold mean = 60, sample variance = 12000/4; pooled = 4/11.
        rows = rows_for_units([("F1", True, 1), ("F2", False, 3),
                              ("F3", True, 2), ("F4", False, 4), ("F5", True, 1)])
        run = evaluation.evaluate(rows, ["a", "b"], expected_folds=5)["runs"][0]
        self.assertEqual(run["fold_summary"]["accuracy"]["mean"], 60)
        self.assertAlmostEqual(run["fold_summary"]["accuracy"]["sample_std"], math.sqrt(3000))
        self.assertEqual(run["fold_summary"]["accuracy"]["n_folds"], 5)
        self.assertAlmostEqual(run["pooled"]["accuracy"], 400 / 11)
        # Fixed two-class denominator: a perfect all-a fold has macro F1=50.
        self.assertEqual(run["fold_summary"]["macro_f1"]["mean"], 30)
        self.assertEqual(set(run["fold_summary"]), set(evaluation.SCORE_KEYS))

    def test_lopo_reports_port_sd_not_fold_sd(self):
        rows = rows_for_units([("A", True, 1), ("B", False, 3)])
        rows = [dict(row, fold="one") for row in rows]
        out = evaluation.evaluate(rows, ["a", "b"], "lopo")
        run = out["runs"][0]
        self.assertEqual(run["port_summary"]["accuracy"]["mean"], 50)
        self.assertAlmostEqual(run["port_summary"]["accuracy"]["sample_std"], math.sqrt(5000))
        self.assertEqual(run["port_summary"]["accuracy"]["n_ports"], 2)
        self.assertEqual(run["pooled"]["accuracy"], 25)
        self.assertIsNone(out["seed_summary"]["m"]["accuracy"]["sample_std"])

    def test_seed_sd_is_separate_from_within_run_sd(self):
        first = rows_for_units([("A", True, 1), ("B", False, 3)])
        second = [dict(row, seed="43", pred_label="a") for row in first]
        out = evaluation.evaluate(first + second, ["a", "b"])
        self.assertEqual(out["seed_summary"]["m"]["accuracy"]["n_seeds"], 2)
        self.assertEqual(out["seed_summary"]["m"]["accuracy"]["mean"], 75)
        self.assertAlmostEqual(out["seed_summary"]["m"]["accuracy"]["sample_std"], math.sqrt(1250))
        self.assertEqual(out["runs"][1]["fold_summary"]["accuracy"]["sample_std"], 0)

    def test_lopo_repeat_then_port_aggregation_keeps_sd_axes_separate(self):
        # Seed 42: A=0, B=100; seed 43: A=50, B=100.
        # Port means are A=25 and B=100; their sample SD=sqrt(2812.5).
        # The seed-level port means are 50 and 75; their SD=sqrt(312.5).
        first = rows_for_units([("A", False, 4), ("B", True, 4)])
        second = [dict(row, seed="43", pred_label="a" if
                       row["port"] == "B" or int(row["sample_id"]) < 2 else "b")
                  for row in first]
        out = evaluation.evaluate(first + second, ["a", "b"], "lopo")
        summary = out["lopo_summary"]["m"]
        port_a = summary["ports"]["A"]["accuracy"]
        port_b = summary["ports"]["B"]["accuracy"]
        self.assertEqual(port_a["mean"], 25)
        self.assertEqual(port_a["n_seeds"], 2)
        self.assertAlmostEqual(port_a["sample_std"], math.sqrt(1250))
        self.assertEqual(port_b["mean"], 100)
        self.assertEqual(port_b["sample_std"], 0)
        combined = summary["port_mean_summary"]["accuracy"]
        self.assertEqual(combined["mean"], 62.5)
        self.assertEqual(combined["n_ports"], 2)
        self.assertAlmostEqual(combined["sample_std"], math.sqrt(2812.5))
        self.assertAlmostEqual(out["seed_summary"]["m"]["accuracy"]["sample_std"], math.sqrt(312.5))
        self.assertAlmostEqual(out["runs"][0]["port_summary"]["accuracy"]["sample_std"], math.sqrt(5000))
        # Mean of per-seed F1 is not F1 after pooling repeated predictions.
        self.assertAlmostEqual(summary["ports"]["A"]["macro_f1"]["mean"], 50 / 3)

    def test_absent_classes_keep_fixed_macro_denominator(self):
        score = evaluation.metrics(rows_for_units([("F", True, 2)]), ["a", "b", "c"])
        self.assertEqual(score["accuracy"], 100)
        self.assertAlmostEqual(score["macro_f1"], 100 / 3)
        self.assertEqual(score["macro_class_count"], 3)
        self.assertEqual(score["absent_classes"], ["b", "c"])

    def test_cross_target_vessel_overlap_is_diagnostic_only(self):
        rows = rows_for_units([("A", True, 1), ("B", True, 1)])
        rows[1]["ship_group_id"] = rows[0]["ship_group_id"]
        run = evaluation.evaluate(rows, ["a", "b"], "lopo")["runs"][0]
        self.assertEqual(run["identity_overlap"]["groups"], 1)
        self.assertFalse(run["training_identity_verified"])
        records = [dict(model="m", seed="42", split=port,
                        train_groups=["source"], validation_groups=["monitor"], test_groups=["g0"])
                   for port in ("A", "B")]
        run = evaluation.evaluate(rows, ["a", "b"], "lopo", records)["runs"][0]
        self.assertTrue(run["training_identity_verified"])
        self.assertEqual(run["identity_overlap"]["groups"], 1)

    def test_actual_train_test_identity_leak_is_rejected(self):
        rows = rows_for_units([("A", True, 1)])
        records = [dict(model="m", seed="42", split="A",
                        train_groups=["g0"], validation_groups=[], test_groups=["g0"])]
        with self.assertRaisesRegex(ValueError, "leaks between train"):
            evaluation.evaluate(rows, ["a", "b"], identity_splits=records)

    def test_identity_record_must_cover_the_evaluated_split(self):
        rows = rows_for_units([("A", True, 1)])
        with self.assertRaisesRegex(ValueError, "Missing train/validation/test"):
            evaluation.evaluate(rows, ["a", "b"], identity_splits=[])
        records = [dict(model="m", seed="42", split="A",
                        train_groups=["source"], validation_groups=[], test_groups=["wrong"])]
        with self.assertRaisesRegex(ValueError, "does not match predictions"):
            evaluation.evaluate(rows, ["a", "b"], identity_splits=records)

    def test_expected_fold_count_and_group_isolation(self):
        rows = rows_for_units([("A", True, 1), ("B", False, 1)])
        with self.assertRaisesRegex(ValueError, "Expected 5 held-out folds"):
            evaluation.evaluate(rows, ["a", "b"], expected_folds=5)
        rows[1]["ship_group_id"] = rows[0]["ship_group_id"]
        with self.assertRaisesRegex(ValueError, "multiple outer held-out folds"):
            evaluation.evaluate(rows, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
