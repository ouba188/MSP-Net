"""Count parameters for a declared configuration, separating training-only heads.

Example: python -m tools.count_parameters --osm-dim 68 --availability-dim 5
No published accuracy or FLOPs is inferred from these counts.
"""
import argparse
import json
from mspnet.model import MSPNet
from mspnet.size_estimator import SARSizeEstimator


def count_parameters(osm_dim, availability_dim=0, num_classes=3):
    model = MSPNet(osm_dim, osm_availability_dim=availability_dim, num_classes=num_classes)
    components = {name: sum(p.numel() for p in child.parameters())
                  for name, child in model.named_children()}
    total = sum(components.values())
    alignment = sum(p.numel() for child in (model.fpcsa.visual_projection, model.fpcsa.prior_projection)
                    for p in child.parameters())
    size = sum(p.numel() for p in SARSizeEstimator().parameters())
    return {"osm_continuous_dim": osm_dim, "osm_availability_dim": availability_dim,
            "num_classes": num_classes, "components": components,
            "classifier_registered_parameters": total,
            "training_only_alignment_parameters": alignment,
            "classifier_inference_path_parameters": total-alignment,
            "independent_size_estimator_parameters": size,
            "online_model_parameters_including_size_estimator": total-alignment+size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osm-dim",type=int,required=True)
    parser.add_argument("--availability-dim",type=int,default=0)
    parser.add_argument("--num-classes",type=int,default=3)
    args=parser.parse_args()
    print(json.dumps(count_parameters(args.osm_dim,args.availability_dim,args.num_classes),indent=2))

if __name__=="__main__":main()
