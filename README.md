# XLoc-CXR

Explainability Localization on Chest X-rays: quantitative evaluation of Grad-CAM and its variants against radiologist bounding-box annotations on VinDr-CXR.

## Overview

This repository trains a multi-label ResNet-50 classifier on the VinDr-CXR dataset and evaluates the spatial fidelity of three Class Activation Mapping (CAM) methods:

- **Grad-CAM**, **Grad-CAM++**, **XGradCAM**
- at two network depths: **layer3** and **layer4**
- with four localization metrics: pointing game, energy in box, IoU, normalized distance
- plus statistical controls: untrained model, weight randomization, random maps

### Key results

- Macro AUC of **0.920** (15 classes) on the test set.
- Best pointing game of **0.474** for cardiomegaly (layer4 + Grad-CAM++), but **0.000-0.087** for focal lesions (nodule, pneumothorax, calcification).
- Weight randomization shows that ILD localization under Grad-CAM is an architectural artifact (p = 0.96), while cardiomegaly localization reflects learned features (p = 2.86e-21).
- Recommended default configuration: **layer4 + Grad-CAM++**, with a systematic weight randomization test before any clinical interpretation.

## Repository layout

```
configs/          Experiment configuration (default.yaml)
data/             VinDr-CXR dataset and preprocessing cache
outputs/          Results, checkpoints, figures (results.csv, auc.json, ...)
rapport/          Reports in French and English (LaTeX sources + PDFs)
scripts/          CLI entry points (train, evaluate, analyze, reanalyze, examples)
src/xloc_cxr/     Package source (data, model, metrics, analysis, visualization)
tests/            Unit tests (dataset, metrics, CAM)
```

## Requirements

- Python >= 3.11
- ROCm-capable GPU recommended for training (`jax[rocm]`); CPU works for analysis scripts

## Installation

```bash
git clone https://github.com/5uru/XLoc-CXR.git
cd XLoc-CXR
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data

VinDr-CXR is a public dataset from the VinBigData Kaggle competition.
Download it with the Kaggle CLI and place it under `data/vindr_cxr`:

```bash
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection
```

Preprocessing (DICOM rescale, windowing, resize to 224x224) is cached once in `data/cache_224` as `.npy` files.

## Usage

All commands accept `--config configs/default.yaml` (default).

```bash
# Train (30 epochs, batch size 64, seed 42)
python scripts/train.py --epochs 30

# Evaluate CAM localization metrics against ground-truth boxes
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pkl

# Statistical analysis: controls, randomization tests, per-pathology breakdown
python scripts/analyze.py --checkpoint outputs/checkpoints/best_model.pkl

# Post-hoc reanalysis of results.csv (no dataset required)
python scripts/reanalyze.py

# Generate CAM success/failure example figures
python scripts/examples.py --checkpoint outputs/checkpoints/best_model.pkl
```

## Outputs

| File | Content |
|---|---|
| `outputs/results.csv` | Per image, layer, method, class and metric scores |
| `outputs/results_per_class.csv` | Per-pathology aggregates |
| `outputs/auc.json` | Per-class and macro AUC |
| `outputs/controls.json` | Control conditions (untrained, randomized, random maps) |
| `outputs/reanalysis.json` | Paired Wilcoxon tests and bootstrap CIs |
| `outputs/figures/` | All report figures |

## Reports

Two versions of the evaluation report are provided in `rapport/`:

- `rapport.pdf` (French)
- `rapport_en.pdf` (English)


## Tests

```bash
pytest tests/
```

## License

Apache License 2.0. VinDr-CXR is distributed under its own original license.
