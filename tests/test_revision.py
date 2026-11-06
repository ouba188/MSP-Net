import unittest
import importlib.util
import math
import tempfile
import warnings
from pathlib import Path
import numpy as np
import torch
from mspnet import (
    MSPNet,
    SARSizeEstimator,
    OSMPriorEncoder,
    PositionPriorEncoder,
    ScalePriorEncoder,
    frequency_vca_loss,
    load_png_pair,
    mspnet_loss,
)
from mspnet.fpcsa import FPCSA, FrequencyDecomposer

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).parents[1] / 'tools/evaluate_predictions.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
torch.set_num_threads(2)


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_backbone_two_paths(self):
        model = MSPNet(8).eval()
        with torch.no_grad():
            frequency_map, visual = model.visual(torch.rand(2, 3, 224, 224))
        self.assertEqual(tuple(frequency_map.shape), (2, 256, 14, 14))
        self.assertEqual(tuple(visual.shape), (2, 512))

    def test_frequency_masks_and_constant_dc(self):
        module = FrequencyDecomposer(2, 8)
        masks = module.masks(14, 14)
        self.assertEqual(masks.sum((1, 2)).tolist(), [13, 84, 99])
        self.assertTrue(torch.all(masks.sum(0) == 1))
        descriptor = module.descriptors(torch.ones(1, 2, 14, 14))
        self.assertTrue(torch.allclose(descriptor[0, 0], torch.full((2,), torch.log1p(torch.tensor(196.)).item() / 13), atol=1e-6))
        self.assertTrue(torch.allclose(descriptor[0, 1:], torch.zeros(2, 2), atol=1e-6))

    def test_query_controls_preserve_capacity(self):
        modules = [FrequencyDecomposer(256, 512, query_mode=q) for q in FrequencyDecomposer.modes]
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in modules}), 1)
        for m in modules:
            self.assertEqual(tuple(m(torch.rand(2, 256, 14, 14)).shape), (2, 3, 512))

    def test_null_only_is_exact_identity_with_nonzero_film(self):
        model = FPCSA(feature_channels=4, token_dim=16, active_sources=('null',))
        torch.nn.init.normal_(model.film[-1].weight)
        torch.nn.init.normal_(model.film[-1].bias)
        visual = torch.randn(2, 16)
        out = model(torch.randn(2, 4, 14, 14), visual, {'null': torch.randn(2, 16)})
        # Softmax of three logits sums to 1 +/- one ulp; the encoder renormalises
        # band_weights, so ``1 - null_mass`` is either exactly 0 or within a
        # single float32 epsilon and the residual FiLM term cannot move.
        self.assertLessEqual((1 - out.null_mass).abs().max().item(), 2 * torch.finfo(visual.dtype).eps)
        self.assertTrue(torch.allclose(out.feature, visual, atol=1e-6))
        self.assertTrue(torch.allclose(out.null_mass, torch.ones(2, 1)))

    def test_band_weights_are_renormalised(self):
        module = FPCSA(feature_channels=4, token_dim=16)
        out = module(torch.rand(3, 4, 14, 14), torch.randn(3, 16),
                     {n: torch.randn(3, 16) for n in module.source_order})
        ones = torch.ones(3)
        self.assertTrue(torch.allclose(out.band_weights.sum(-1), ones, rtol=0, atol=1e-7))
        self.assertTrue(torch.allclose(out.routing.sum((1, 2)), ones, rtol=0, atol=1e-6))

    def test_per_head_reporting_matches_head_average(self):
        module = FPCSA(feature_channels=4, token_dim=16, num_heads=4)
        out = module(torch.rand(2, 4, 14, 14), torch.randn(2, 16),
                     {n: torch.randn(2, 16) for n in module.source_order})
        self.assertEqual(tuple(out.attention_per_head.shape), (2, 4, 3, 4))
        self.assertEqual(tuple(out.null_mass_per_head.shape), (2, 4))
        self.assertTrue(torch.allclose(out.attention_per_head.mean(1), out.attention, atol=1e-6))
        self.assertTrue(torch.allclose(out.null_mass_per_head.mean(1, keepdim=True), out.null_mass, atol=1e-6))
        # Heads must not be collapsed inside the attention: at least one head
        # pair must differ, otherwise the head axis would be cosmetic.
        self.assertFalse(torch.allclose(out.attention_per_head[:, 0], out.attention_per_head[:, 1]))

    def test_full_forward_backward_and_frozen_size_inputs(self):
        model = MSPNet(8)
        size = torch.tensor([[5., 3., 0., 0.], [5.2, 3.1, .2, .1]], requires_grad=True)
        # The scale descriptor statistics must be fitted on out-of-fold training
        # predictions; without this call the encoder warns.
        model.scale_encoder.set_descriptor_statistics(size.detach())
        out = model(torch.rand(2, 3, 224, 224), size, torch.rand(2, 2), torch.rand(2, 3), torch.rand(2, 8))
        self.assertEqual(tuple(out['attention'].shape), (2, 3, 4))
        self.assertTrue(torch.allclose(out['routing'].sum((1, 2)), torch.ones(2)))
        loss, _ = mspnet_loss(out['logits'], torch.tensor([0, 1]), out['visual_alignment'], out['prior_alignment'])
        loss.backward()
        self.assertIsNone(size.grad)
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(model.visual.layer3[0].conv1.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.fpcsa.query.weight.grad.abs().sum().item(), 0)

    def test_null_value_participates_in_mixed_source_condition(self):
        module = FPCSA(feature_channels=1, token_dim=4, num_heads=1,
                       active_sources=('scale', 'null'),
                       compatibility_bias=False, source_gating=False)
        with torch.no_grad():
            module.query.weight.zero_()
            module.query.bias.zero_()
            module.value.weight.copy_(torch.eye(4))
            module.output.weight.copy_(torch.eye(4))
            module.output.bias.zero_()
        null = torch.tensor([[2., 4., 6., 8.]], requires_grad=True)
        out = module(torch.ones(1, 1, 14, 14), torch.ones(1, 4),
                     {'scale': torch.zeros(1, 4), 'null': null})
        self.assertTrue(torch.allclose(out.prior_mix, torch.tensor([[1., 2., 3., 4.]])))
        self.assertTrue(torch.allclose(out.null_mass, torch.tensor([[.5]])))
        out.prior_mix.sum().backward()
        self.assertTrue(torch.allclose(null.grad, torch.full((1, 4), .5)))

    def test_multihead_attention_keeps_head_specific_weights(self):
        class FixedQueries(torch.nn.Module):
            def forward(self, x):
                return x.new_tensor([4., 0., 4., 0.]).reshape(1, 1, 4).expand(x.shape[0], 3, -1)
        model = FPCSA(feature_channels=1, token_dim=4, num_heads=2, temperature=1., active_sources=('scale', 'osm'), source_gating=False, compatibility_bias=False)
        model.frequency = FixedQueries()
        with torch.no_grad():
            for projection in (model.query, model.key, model.value):
                projection.weight.copy_(torch.eye(4))
                if projection.bias is not None: projection.bias.zero_()
        captured = []
        handle = model.output.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach()))
        with torch.no_grad():
            model(torch.ones(1, 1, 14, 14), torch.ones(1, 4), {'scale': torch.tensor([[1.,0.,0.,1.]]), 'osm': torch.tensor([[0.,1.,1.,0.]])})
        handle.remove()
        p = torch.sigmoid(torch.tensor(4. / 2**.5)).item()
        expected = torch.tensor([p, 1-p, p, 1-p]).reshape(1,1,4).expand(1,3,4)
        self.assertTrue(torch.allclose(captured[0], expected, atol=1e-6))

    def test_alignment_heads_skipped_at_inference(self):
        module = FPCSA(feature_channels=4, token_dim=16).eval()
        out = module(torch.rand(2, 4, 14, 14), torch.randn(2,16), {n:torch.randn(2,16) for n in module.source_order})
        self.assertIsNone(out.visual_alignment)
        self.assertIsNone(out.prior_alignment)

    def test_eval_classification_loss_and_explicit_alignment(self):
        module = FPCSA(feature_channels=4, token_dim=16).eval()
        inputs = (torch.rand(2, 4, 14, 14), torch.randn(2, 16),
                  {n: torch.randn(2, 16) for n in module.source_order})
        out = module(*inputs)
        logits, labels = torch.randn(2, 3), torch.tensor([0, 1])
        loss, parts = mspnet_loss(logits, labels, out.visual_alignment,
                                 out.prior_alignment, lambda_v=0)
        expected = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=.08)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(parts['freq_vca'].item(), 0)
        with self.assertRaisesRegex(ValueError, 'return_alignment=True'):
            mspnet_loss(logits, labels, out.visual_alignment, out.prior_alignment)
        aligned = module(*inputs, return_alignment=True)
        joint_loss, _ = mspnet_loss(logits, labels, aligned.visual_alignment, aligned.prior_alignment)
        self.assertTrue(torch.isfinite(joint_loss))

    def test_osm_can_suppress_all_channels(self):
        module = OSMPriorEncoder(8, token_dim=16)
        torch.nn.init.zeros_(module.feature_gate[0].weight)
        torch.nn.init.constant_(module.feature_gate[0].bias, -20.)
        self.assertLess(module(torch.randn(2, 8)).abs().max().item(), 1e-7)

    def test_spherical_encoding_dateline(self):
        module = PositionPriorEncoder(token_dim=16)
        local = torch.zeros(1, 3)
        self.assertTrue(torch.allclose(module(torch.tensor([[180., 30.]]), local), module(torch.tensor([[-180., 30.]]), local), atol=1e-5))

    def test_position_gate_weighted_sum(self):
        module = PositionPriorEncoder(token_dim=4)
        with torch.no_grad():
            for branch, constant in ((module.global_mlp, 2.), (module.local_mlp, -2.)):
                branch[-1].weight.zero_()
                branch[-1].bias.fill_(constant)
            module.route.weight.zero_()
            module.route.bias.fill_(torch.log(torch.tensor(3.)).item())
            module.output.weight.copy_(torch.eye(4))
        actual = module(torch.tensor([[120., 30.]]), torch.zeros(1, 3))
        # alpha=3/4: (3/4)*2 + (1/4)*(-2) = 1 in every channel.
        self.assertTrue(torch.allclose(actual, torch.ones(1, 4)))

    def test_size_branch_independent_positive_and_finite(self):
        classifier, estimator = MSPNet(8), SARSizeEstimator()
        self.assertFalse({id(p) for p in classifier.parameters()} & {id(p) for p in estimator.parameters()})
        result = estimator(torch.rand(2, 3, 224, 224))
        self.assertEqual(tuple(result.shape), (2, 4))
        self.assertTrue((result[:, :2] > 0).all())
        self.assertTrue(torch.isfinite(result).all())

    def test_scale_confidence_is_bounded(self):
        encoder = ScalePriorEncoder(token_dim=32, confidence_logvar_bound=6.0)
        encoder.set_descriptor_statistics(torch.randn(32, 4))
        bound = math.exp(3.0)  # exp(+/- 0.5 * 6) in each confidence channel
        confidence = {}
        for logvar in (-12.0, -6.0, 0.0, 3.0, 12.0):
            stats = torch.tensor([[5.0, 3.0, logvar, logvar]])
            value = encoder.confidence(ScalePriorEncoder.descriptor(stats)).item()
            confidence[logvar] = value
            self.assertLessEqual(value, bound * (1 + 1e-6))
            self.assertGreaterEqual(value, 1.0 / bound * (1 - 1e-6))
            token = encoder(stats)
            self.assertTrue(torch.isfinite(token).all())
        # Unclamped, logvar=-12 would give exp(6)=403; the bound caps it at exp(3).
        self.assertAlmostEqual(confidence[-12.0], bound, places=5)
        self.assertAlmostEqual(confidence[-6.0], bound, places=5)
        # Ordering is preserved: lower predicted variance never means less weight.
        ordered = [confidence[v] for v in (-12.0, -6.0, 0.0, 3.0, 12.0)]
        self.assertEqual(ordered, sorted(ordered, reverse=True))

    def test_scale_encoder_warns_when_statistics_are_unfitted(self):
        encoder = ScalePriorEncoder(token_dim=8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            encoder(torch.tensor([[5.0, 3.0, 0.0, 0.0]]))
        self.assertTrue(any('never fitted' in str(item.message) for item in caught))
        encoder.set_descriptor_statistics(torch.randn(16, 4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            encoder(torch.tensor([[5.0, 3.0, 0.0, 0.0]]))
        self.assertFalse(any('never fitted' in str(item.message) for item in caught))

    def test_frequency_vca_loss_warns_for_single_sample(self):
        aligned = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            loss = frequency_vca_loss(aligned, aligned.clone())
        self.assertEqual(loss.item(), 0)
        self.assertTrue(any('at least two samples' in str(item.message) for item in caught))

    def test_png_pair_loads_writable_and_silent(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('vv', 'vh'):
                Image.fromarray(np.full((64, 64), 40, dtype=np.uint8)).save(root / f'{name}.png')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                tensor = load_png_pair(root / 'vv.png', root / 'vh.png')
            self.assertFalse([c for c in caught if 'writable' in str(c.message).lower()])
            self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
            self.assertTrue(tensor.is_contiguous())
            self.assertTrue(tensor.requires_grad is False)


class EvaluationTests(unittest.TestCase):
    def rows(self):
        return [dict(model='m', seed='42', sample_id=str(i), ship_group_id=str(i), port=p, fold=p, true_label=t, pred_label=y)
                for i, (p, t, y) in enumerate([('A','a','a'), ('A','b','a'), ('B','a','a'), ('B','b','b')])]

    def test_metrics_count_and_macro(self):
        score = evaluation.metrics(self.rows(), ['a','b'])
        self.assertEqual(score['correct'], 3)
        self.assertEqual(score['accuracy'], 75.)
        self.assertAlmostEqual(score['macro_f1'], (0.8 + 2/3) * 50)

    def test_missing_std_is_not_invented(self):
        out = evaluation.evaluate(self.rows(), ['a','b'])
        self.assertIsNone(out['seed_summary']['m']['accuracy']['sample_std'])
        self.assertAlmostEqual(evaluation.summarize([70,80])['sample_std'], 50**.5)

    def test_lopo_aggregation(self):
        out = evaluation.evaluate(self.rows(), ['a','b'], 'lopo')
        self.assertEqual(out['runs'][0]['port_summary']['accuracy']['mean'], 75.)

    def test_duplicate_and_group_leak_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError): evaluation.evaluate(rows + [rows[0]], ['a','b'])
        rows[2]['ship_group_id'] = rows[0]['ship_group_id']
        with self.assertRaises(ValueError): evaluation.evaluate(rows, ['a','b'])
        with self.assertRaises(ValueError): evaluation.assert_identity_disjoint(['x'], ['y'], ['x'])

    def test_comparison_alignment_rejected(self):
        rows = self.rows()
        other = [dict(r, model='other') for r in rows]
        other[0]['true_label'] = 'b'
        with self.assertRaises(ValueError):
            evaluation.evaluate(rows + other, ['a','b'])

    def test_macro_uses_fixed_class_set(self):
        rows = [dict(model='m', seed='42', sample_id=f's{i}', ship_group_id=f'g{i}',
                     port='small', fold='small', true_label='a', pred_label='a') for i in range(5)]
        score = evaluation.metrics(rows, ['a', 'b', 'c'])
        self.assertEqual(score['accuracy'], 100.)
        self.assertAlmostEqual(score['macro_f1'], 100. / 3)
        self.assertEqual(score['macro_class_count'], 3)
        self.assertEqual(score['absent_classes'], ['b', 'c'])

    def test_supported_but_never_predicted_class_still_scores_zero(self):
        rows = [dict(model='m', seed='42', sample_id=f's{i}', ship_group_id=f'g{i}',
                     port='A', fold='A', true_label='a', pred_label='a') for i in range(4)]
        rows.append(dict(model='m', seed='42', sample_id='s9', ship_group_id='g9',
                         port='A', fold='A', true_label='b', pred_label='a'))
        score = evaluation.metrics(rows, ['a', 'b'])
        self.assertEqual(score['macro_class_count'], 2)
        self.assertEqual(score['absent_classes'], [])
        # Class a: precision 4/5, recall 1 -> f1 0.889. Class b: never predicted
        # -> f1 0. Macro therefore sits between the two, not at 100.
        self.assertAlmostEqual(score['macro_f1'], 100 * (2 * 4 / 9) / 2, places=6)

    def test_lopo_cross_port_identity_recorded_not_assumed_leak(self):
        rows = [
            dict(model='m', seed='42', sample_id='s1', ship_group_id='g1', port='A', fold='A',
                 true_label='a', pred_label='a'),
            dict(model='m', seed='42', sample_id='s2', ship_group_id='g1', port='B', fold='B',
                 true_label='a', pred_label='a'),
        ]
        out = evaluation.evaluate(rows, ['a', 'b'], 'lopo')
        self.assertFalse(out['runs'][0]['training_identity_verified'])
        self.assertEqual(out['runs'][0]['identity_overlap']['groups'], 1)
        # The same vessel is harmless in the pooled protocol, where the split
        # unit is the outer fold rather than the port.
        pooled = [dict(row, fold='F1') for row in rows]
        self.assertNotIn('identity_overlap', evaluation.evaluate(pooled, ['a', 'b'])['runs'][0])

if __name__ == '__main__':
    unittest.main()
