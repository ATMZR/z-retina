# z-retina

CAFormer-B36 + ML-Decoder for **multi-label fundus** classification: training, **Table S1** evaluation (8× TTA, val F1-max thresholds), and **patient-level splits** without train/val leakage.

Dependencies and CLI are managed with **[Poetry](https://python-poetry.org/)**.

## Install

```bash
cd z-retina
poetry install
```

(Optional dev tools: `pytest`, Jupyter.)

## Local weights (not in Git)

If you copied **`checkpoints/best.pth`** locally (~1.1 GB), use it for eval; the whole **`checkpoints/`** directory is **gitignored** — do not commit weights.

## Train

Edit paths in `configs/default.yaml`, then:

```bash
poetry run z-retina-train --config configs/default.yaml
```

### Patient split backends (`patient_split_backend` in YAML)

| Value | Behaviour |
|-------|-----------|
| **`research`** (default) | Same pipeline as **`new_eyes`**: optional **dHash** (+ ORB/RANSAC on CLAHE vessels) for missing `patient_id`, then **`MultilabelStratifiedShuffleSplit`** over patient groups. |
| **`group_shuffle`** | Alternative: **pHash** + Frangi/ORB + **`GroupShuffleSplit`** (simpler label stratification). |

If `split_json` is missing and `auto_generate_patient_split: true`, the chosen backend writes that JSON before training.

## Evaluate (Table S1 protocol)

```bash
poetry run z-retina-eval --config configs/default.yaml --checkpoint checkpoints/best.pth --table_s1
```

Optional: `--no_tta`, `--out_json metrics.json`.

## Generate split only (CLI)

Same backends as `z-retina-split` (wraps the pseudo-patient module CLI):

```bash
poetry run z-retina-split --merged_json ./data/merged.json --images_root ./data/eyes/dataset --out_json ./splits/patient_split.json --backend research
```

Use `--backend group_shuffle` for the pHash variant. `--no_infer` skips similarity when all rows have `patient_id`.

## Project layout

```
checkpoints/             # optional local .pth (gitignored — do not commit)
configs/default.yaml     # training / paths / patient_split_backend
src/z_retina/
  dataset.py             # EyeDataset, preprocessing, augmentations
  model.py               # CAFormer + ML-Decoder
  asl.py                 # asymmetric loss
  patient_split.py       # new_eyes-equivalent split (dHash + MLSS)
  pseudo_patient.py      # pHash + GroupShuffle alternative + CLI
  splits.py              # dispatches backend from config (for train)
  evaluate.py            # metrics + Table S1
  apps/                  # Poetry console entrypoints
  third_party/ml_decoder/  # vendored ML-Decoder
tests/test_smoke.py
reproduce_tableS1.ipynb  # Colab-oriented (Poetry install + eval)
```

## Tests

```bash
poetry run pytest -v
```

## Licence note

`third_party/ml_decoder` is third-party code; keep its licence terms if you redistribute.
