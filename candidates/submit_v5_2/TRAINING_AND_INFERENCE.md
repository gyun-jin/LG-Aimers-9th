# TRAINING_AND_INFERENCE

## Requirements

```bash
pip install -r requirements.txt
```

## Required Data For Training

Place official files outside this Git package:

```text
data/train.csv
data/trackman_history.csv
```

The mapping file is included:

```text
features/pitcher_id_mapping_clean.csv
```

## Training Commands

Candidate experiments:

```bash
python train.py
```

Full validation:

```bash
python full_validation.py
```

Build final submit package:

```bash
python build_submit.py
```

## Inference Command

In an extracted submit environment:

```bash
python script.py
```

The script searches:

```text
./open/test.csv
./open/sample_submission.csv
./data/test.csv
./data/sample_submission.csv
```

It writes:

```text
output/submission.csv
```

## Evaluation Server

The competition server installs `requirements.txt`, extracts `submit.zip`, adds test data under `open/` or `data/`, and runs `script.py`.

## Rebuilding submit.zip

`build_submit.py` trains the selected `tm_metadata_physical_pitchmix` model, stores model artifacts, copies the inference script, and writes `submit.zip`.

Expected submit members:

```text
model/
model/final_model.joblib
model/calibration_model.joblib
model/feature_schema.json
model/ensemble_config.json
script.py
requirements.txt
```
