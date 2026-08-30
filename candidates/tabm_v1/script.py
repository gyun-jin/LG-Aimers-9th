from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ID_COL, TARGET_COL, Preprocessor
from tabm_model import make_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("./data/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("./output/submission.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "model")
    parser.add_argument("--batch-size", type=int, default=16_384)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.batch_size > 2048:
        args.batch_size = 2048

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    if ID_COL not in frame:
        raise ValueError(f"Input CSV does not contain {ID_COL}")
    prep = Preprocessor.load(args.model_dir / "preprocessor.json")
    x_num, x_cat = prep.transform(frame)
    checkpoint = torch.load(args.model_dir / "tabm.pt", map_location="cpu", weights_only=True)
    model = make_model(
        int(checkpoint["n_num_features"]),
        list(checkpoint["cat_cardinalities"]),
        list(checkpoint["bins"]),
        dict(checkpoint["model_config"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    predictions = []
    for start in range(0, len(frame), args.batch_size):
        xn = torch.from_numpy(x_num[start : start + args.batch_size]).to(device)
        xc = torch.from_numpy(x_cat[start : start + args.batch_size]).to(device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(xn, xc).squeeze(-1)
        predictions.append(logits.float().sigmoid().mean(dim=1).cpu().numpy())
    prediction = np.clip(np.concatenate(predictions), 0.0, 1.0)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction detected")

    output = pd.DataFrame({ID_COL: frame[ID_COL].to_numpy(), TARGET_COL: prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"device={device} rows={len(output):,} saved={args.output}")


if __name__ == "__main__":
    main()
