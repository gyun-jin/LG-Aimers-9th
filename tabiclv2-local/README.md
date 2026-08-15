# TabICLv2 local runner

This runner avoids loading all 1,475,092 training rows into memory. It keeps an
exact uniform reservoir sample while reading `../data/train.csv` in chunks,
preserves the natural class ratio, and runs TabICLv2 on CPU with disk offload.

## Install

Run these commands from this directory:

```powershell
conda create -n tabiclv2 python=3.12 -y
conda activate tabiclv2
python -m pip install -r requirements.txt
```

The pretrained checkpoint is downloaded on the first run.

## Run

Start with the 1,000-row smoke preset:

```powershell
python train.py --preset smoke
```

Then try the 10,000-row safe preset:

```powershell
python train.py --preset safe
```

The intended local run uses 30,000 context rows and 2,000 validation rows:

```powershell
python train.py --preset full
```

Outputs are written to:

- `model/classifier.pkl`
- `model/metadata.json`
- `output/submission.csv`

By default the model file refers to the checkpoint in the local TabICL cache.
Add `--save-weights` if the pickle must be self-contained (it will be much larger):

```powershell
python train.py --preset full --save-weights
```

After training, rerun prediction without resampling:

```powershell
python predict.py
```

Override sizes only after the safe preset succeeds, for example:

```powershell
python train.py --context-size 48000 --validation-size 2000
```

`kv_cache` and AMP are intentionally disabled to control memory use. The Intel
Arc 140V is not selected because TabICL does not currently document XPU as a
supported backend.
