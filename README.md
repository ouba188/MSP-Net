# MSP-Net Core

This repository contains the compact core implementation of **MSP-Net**, the
frequency-conditioned multi-source-prior framework for SAR ship
classification described in the accompanying manuscript.

## Scope

The release is limited to the manuscript-facing model components:

- VV/VH/polarization-contrast three-channel construction;
- scale, port-position, OSM-semantic, and learnable null prior encoders;
- frequency decomposition and frequency-conditioned prior cross-attention
  (FPCSA), including source gating, null-prior-aware modulation, and Freq-VCA;
- the Stage-1 heteroscedastic size-estimation loss and the Stage-2
  classification/Freq-VCA loss; and
- a small CPU smoke test.

Cloud training launchers, checkpoints, private experiment logs, and
institution-specific data preparation scripts are intentionally not included.
The Stage-1 SAR size estimator is trained independently as described in the
paper.  The released classifier consumes its frozen predictions
`[mu_L, mu_W, logvar_L, logvar_W]` and implements the Stage-2 FPCSA
classification path.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test checks tensor shapes, source-attention normalization, finite
outputs, and a backward pass.  A GPU is not required for this check.

## Stage protocol

1. Train a SAR size estimator with `heteroscedastic_size_loss` using log-domain
   AIS length and width targets.
2. Within each outer classification fold, generate out-of-fold size
   predictions for the outer training samples and fit the estimator on the
   complete outer training set for the held-out samples.  Keep the estimator
   frozen during Stage 2.
3. Construct the SAR input with `build_three_channel` and call `MSPNet` with
   the frozen size statistics, longitude/latitude, port-local position, and
   numeric OSM features.  The returned `routing` tensor has shape
   `[batch, 3 frequency bands, 4 prior sources]` in the order
   `scale, position, osm, null`.

The full cloud experiment project is larger than this public core release.
This repository is intended for auditing and reusing the MSP-Net prior
encoding and FPCSA classification path.

## Companion data

The sanitized seven-port SAR-AIS chips and metadata are released separately:

<https://github.com/ouba188/SAR-AIS-Dataset>

## Attribution and contact

The method is described in the submitted MSP-Net manuscript.  OSM-derived
features in the companion dataset require attribution to OpenStreetMap
contributors.  Please retain the paper citation and the data-source notices
when redistributing derived artifacts.
