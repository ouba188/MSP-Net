import argparse
import json
from pathlib import Path

from mspnet.training import PreparedData, TrainingConfig, run_protocol
from tools.evaluate_predictions import evaluate


def main():
    parser = argparse.ArgumentParser(description='Train the manuscript protocol from prepared arrays; see mspnet.training.')
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--protocol", choices=("all-port", "lopo"))
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, help="Training seed; grouped split_seed stays fixed")
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if args.protocol:
        values["protocol"] = args.protocol
    if args.device:
        values["device"] = args.device
    if args.seed is not None:
        values["seed"] = args.seed
    config = TrainingConfig(**values)
    data = PreparedData.from_npz(args.prepared)
    if data.sar.shape[-2:] != (224, 224):
        raise ValueError("The manuscript backbone expects prepared 224x224 inputs")
    rows = run_protocol(data, config, args.output)
    identity = json.loads((args.output / "identity_splits.json").read_text(encoding="utf-8"))
    summary = evaluate(rows, data.class_names.tolist(), config.protocol,
                       identity_splits=identity, expected_folds=config.folds)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Finished {len(rows)} held-out predictions: {args.output}")


if __name__ == "__main__":
    main()
