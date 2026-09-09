# MSP-Net Core

This repository contains only the core implementation of **MSP-Net**, the
frequency-conditioned multi-source-prior framework for SAR ship classification
described in the accompanying manuscript.

## Included

- `mspnet/sar_input.py`: VV/VH/polarization-contrast input construction;
- `mspnet/priors.py`: scale, position, OSM-semantic, and null-prior encoders;
- `mspnet/fpcsa.py`: frequency decomposition and FPCSA fusion;
- `mspnet/model.py`: ResNet-18 visual backbone and MSP-Net classification path;
- `mspnet/losses.py`: Stage-1 size loss and Stage-2 classification/Freq-VCA
  losses.

The Stage-1 SAR size estimator is trained independently as described in the
paper. The released classifier consumes its frozen predictions
`[mu_L, mu_W, logvar_L, logvar_W]`; the estimator architecture and cloud
training pipeline are outside this compact release.

## Minimal use

```bash
pip install -r requirements.txt
```

Construct the three-channel input with `build_three_channel`, then call
`MSPNet` with frozen size statistics, longitude/latitude, port-local position,
and numeric OSM features. The model returns class logits and the FPCSA routing
tensor in the source order `scale, position, osm, null`.

This repository is intended to expose the manuscript's core model logic. It
does not include cloud training launchers, checkpoints, private experiment
logs, or institution-specific data preparation scripts.

## Companion data

The sanitized seven-port SAR-AIS chips and metadata are released separately:

<https://github.com/ouba188/SAR-AIS-Dataset>

## Attribution and contact

The method is described in the submitted MSP-Net manuscript.  OSM-derived
features in the companion dataset require attribution to OpenStreetMap
contributors.  Please retain the paper citation and the data-source notices
when redistributing derived artifacts.
