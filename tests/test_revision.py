import unittest
import importlib.util
from pathlib import Path
import torch
from mspnet import MSPNet, SARSizeEstimator, OSMPriorEncoder, PositionPriorEncoder, mspnet_loss
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
        self.assertTrue(torch.equal(out.feature, visual))
        self.assertTrue(torch.allclose(out.null_mass, torch.ones(2, 1)))

    def test_full_forward_backward_and_frozen_size_inputs(self):
        model = MSPNet(8)
        size = torch.tensor([[5., 3., 0., 0.], [5.2, 3.1, .2, .1]], requires_grad=True)
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
        self.assertIsNone(out['summary']['m']['accuracy']['sample_std'])
        self.assertAlmostEqual(evaluation.summarize([70,80])['sample_std'], 50**.5)

    def test_lopo_aggregation(self):
        out = evaluation.evaluate(self.rows(), ['a','b'], 'lopo')
        self.assertEqual(out['runs'][0]['port_macro']['accuracy'], 75.)

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
        with self.assertRaises(ValueError): evaluation.evaluate(rows + other, ['a','b'])


if __name__ == '__main__':
    unittest.main()
