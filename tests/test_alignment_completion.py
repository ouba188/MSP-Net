import itertools
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from mspnet.fpcsa import FPCSA
from mspnet.model import MSPNet
from mspnet.priors import ScalePriorEncoder
from mspnet.sar_input import load_png_pair
from mspnet.training import PreparedData, TrainingConfig, train_classifier

torch.set_num_threads(2)

class AlignmentCompletionTests(unittest.TestCase):
    def test_all_band_subsets_have_real_active_queries(self):
        torch.manual_seed(12)
        for count in (1, 2, 3):
            for bands in itertools.combinations(("low", "mid", "high"), count):
                m = FPCSA(feature_channels=4, token_dim=16, active_bands=bands)
                with torch.no_grad():
                    m.film[-1].weight.normal_()
                out = m(torch.rand(2, 4, 14, 14), torch.rand(2, 16),
                        {n: torch.rand(2,16) for n in m.source_order})
                self.assertEqual(out.frequency_tokens.shape, (2,count,16))
                self.assertEqual(m.visual_projection[0].in_features, count*16)
                self.assertEqual(out.band_names, bands)
                self.assertTrue(torch.allclose(out.attention.sum(-1),torch.ones(2,count)))
                self.assertTrue(torch.allclose(out.band_weights.sum(-1),torch.ones(2)))
                out.feature.square().sum().backward()
                self.assertGreater(m.frequency.projection.weight.grad.abs().sum().item(),0)

    def test_scale_statistics_survive_checkpoint(self):
        stats=torch.tensor([[5.,3.,-12.,12.],[5.3,3.2,1.,-2.]])
        a=ScalePriorEncoder(16);a.set_descriptor_statistics(stats)
        b=ScalePriorEncoder(16);b.load_state_dict(a.state_dict())
        with warnings.catch_warnings(record=True) as caught:
            actual=b(stats)
        self.assertFalse(caught)
        self.assertTrue(torch.equal(actual,a(stats)))

        self.assertAlmostEqual(a.descriptor(stats)[0,4].item(),np.exp(6),places=4)
        self.assertAlmostEqual(a.confidence(a.descriptor(stats))[0].item(),1.,places=6)

    def test_png_scaling_does_not_depend_on_sample_maximum(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"p.png";Image.fromarray(np.ones((64,64),dtype=np.uint8)).save(p)
            x=load_png_pair(p,p,output_size=None)
            self.assertAlmostEqual(x[0,0,0,0].item(),np.log1p(1/255),places=7)

    def test_real_fusion_baselines_and_backward(self):
        for kind in ("visual","concat","film","vanilla","fpcsa"):
            m=MSPNet(68,osm_availability_dim=5,fusion_kind=kind)
            size=torch.tensor([[5.,3.,0.,0.],[5.2,3.1,.2,.1]])
            if hasattr(m,"scale_encoder"):m.scale_encoder.set_descriptor_statistics(size)
            out=m(torch.rand(2,3,224,224),size,torch.tensor([[120.,30.],[121.,31.]]),
                  torch.randn(2,3),torch.randn(2,68),torch.ones(2,5))
            self.assertEqual(out["logits"].shape,(2,3))
            torch.nn.functional.cross_entropy(out["logits"],torch.tensor([0,1])).backward()
            self.assertGreater(m.classifier.weight.grad.abs().sum().item(),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_real_classifier_training_and_reload(self):
        n=6;rng=np.random.default_rng(3)
        d=PreparedData(sar=rng.random((n,3,224,224),dtype=np.float32),lonlat=np.tile([120.,30.],(n,1)),
            local_position=rng.normal(size=(n,3)),osm=rng.random((n,68)),osm_availability=np.ones((n,5)),
            osm_availability_index=np.array([i for i in range(5) for _ in range(13)]+[-1]*3),
            osm_feature_names=np.array([str(i) for i in range(68)]),osm_availability_names=np.array(list("abcde")),
            labels=np.array([0,1,2,0,1,2]),class_names=np.array(["a","b","c"]),size_targets=np.tile([100.,20.],(n,1)),
            sample_ids=np.array([str(i) for i in range(n)]),group_ids=np.array([str(i) for i in range(n)]),
            ports=np.array(["p"]*n),preprocessing_json='{"synthetic":true}')
        stats=np.tile([5.,3.,0.,0.],(n,1)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            m,local,osm=train_classifier(d,np.arange(4),np.arange(4,6),stats,
                TrainingConfig(epochs=2,patience=2,batch_size=2),MSPNet,Path(temp))
            self.assertTrue((Path(temp)/"classifier.pt").exists())
            self.assertTrue(bool(m.scale_encoder.statistics_fitted))
            self.assertTrue(torch.isfinite(m.fpcsa.film[-1].weight).all())
            self.assertGreater(m.fpcsa.film[-1].weight.abs().sum().item(),0)

if __name__ == "__main__":unittest.main()
