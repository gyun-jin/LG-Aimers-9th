# TRAINING AND INFERENCE

## Requirements

Install dependencies from:

```bash
pip install -r requirements.txt
```

## Data Placement

The original workspace expected:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/trackman_history.csv
hand_cleaning_analysis/train_hand_trackman_clean.csv
train_trackman/pitcher_id_mapping_clean.csv
```

Official large CSVs and the cleaned train CSV are not committed.

## Training

The v5-3 build script is:

```bash
python build_submit.py
```

It builds OOF validation, fits the final CatBoost and LightGBM models, saves `model/`, and creates `submit.zip`.

## Inference

For local inference from an extracted submission:

```bash
python script.py
```

The script reads `data/test.csv` and `data/sample_submission.csv`, then writes `output/submission.csv`.

## Competition Runner

The evaluation server extracts `submit.zip`, installs `requirements.txt`, runs `script.py`, and reads `output/submission.csv`.

## Recreating submit.zip

`build_submit.py` writes the archive with this top-level structure:

```text
model/
script.py
requirements.txt
```
