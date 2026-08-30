from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import ID_COL, TARGET_COL, Preprocessor
from tabm_model import MODEL_CONFIG, compute_numerical_bins, make_model


SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x_num, x_cat, y, batch_size, shuffle, device):
    dataset = TensorDataset(
        torch.from_numpy(x_num),
        torch.from_numpy(x_cat),
        torch.from_numpy(y.astype(np.float32, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


@torch.inference_mode()
def predict(model, x_num, x_cat, device, batch_size=16_384) -> np.ndarray:
    model.eval()
    outputs = []
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    for start in range(0, len(x_num), batch_size):
        xn = torch.from_numpy(x_num[start : start + batch_size]).to(device, non_blocking=True)
        xc = torch.from_numpy(x_cat[start : start + batch_size]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(xn, xc).squeeze(-1)
        # Official TabM guidance: average probabilities, not logits.
        outputs.append(logits.float().sigmoid().mean(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def train_epochs(
    model,
    train_data,
    device,
    *,
    batch_size,
    max_epochs,
    patience,
    validation_data=None,
):
    x_num, x_cat, y = train_data
    loader = make_loader(x_num, x_cat, y, batch_size, True, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    # torch.cuda.amp.GradScaler also works on older PyTorch releases used by some
    # teammates; it is a no-op for CPU and bfloat16 runs.
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype is torch.float16
    )

    best_state = None
    best_epoch = -1
    best_brier = math.inf
    remaining = patience
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        started = time.time()
        for xn, xc, target in loader:
            xn = xn.to(device, non_blocking=True)
            xc = xc.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                logits = model(xn, xc).squeeze(-1)
                # Each of the k members is optimized independently (loss before averaging).
                loss = F.binary_cross_entropy_with_logits(
                    logits, target[:, None].expand_as(logits)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(target)
            seen += len(target)

        message = f"epoch={epoch + 1:02d} loss={running_loss / seen:.6f} time={time.time()-started:.0f}s"
        if validation_data is None:
            print(message, flush=True)
            continue

        val_num, val_cat, val_y = validation_data
        val_pred = predict(model, val_num, val_cat, device)
        brier = float(np.mean((val_pred - val_y) ** 2))
        improved = brier < best_brier - 1e-7
        print(f"{'*' if improved else ' '} {message} val_brier={brier:.6f}", flush=True)
        if improved:
            best_brier = brier
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            remaining = patience
        else:
            remaining -= 1
            if remaining <= 0:
                break

    if validation_data is None:
        best_epoch = max_epochs - 1
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert best_state is not None
    return best_state, best_epoch, best_brier


def parse_args():
    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=base / "data" / "train.csv")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "model")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--skip-final-refit", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 5,000 rows per season, two epochs, and skip final refit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.batch_size > 1024:
        args.batch_size = 1024
    print(f"device={device} batch_size={args.batch_size}", flush=True)

    frame = pd.read_csv(args.train, encoding="utf-8-sig")
    if TARGET_COL not in frame or ID_COL not in frame:
        raise ValueError("train.csv must contain row_id and control_success")
    if args.smoke:
        frame = frame.groupby("season", sort=False, group_keys=False).head(5_000).copy()
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = 1
        args.skip_final_refit = True
        print(f"smoke mode: {len(frame):,} rows", flush=True)
    input_columns = [c for c in frame.columns if c not in (ID_COL, TARGET_COL)]
    train_mask = frame["season"].to_numpy() <= 2023
    val_mask = frame["season"].to_numpy() == 2024
    if not train_mask.any() or not val_mask.any():
        raise ValueError("Expected 2019-2023 training rows and 2024 validation rows")

    print("[1/2] temporal validation: 2019-2023 -> 2024", flush=True)
    prep = Preprocessor.fit(frame.loc[train_mask], input_columns)
    x_num_train, x_cat_train = prep.transform(frame.loc[train_mask])
    x_num_val, x_cat_val = prep.transform(frame.loc[val_mask])
    y_train = frame.loc[train_mask, TARGET_COL].to_numpy(dtype=np.float32)
    y_val = frame.loc[val_mask, TARGET_COL].to_numpy(dtype=np.float32)
    bins = compute_numerical_bins(x_num_train, seed=SEED)
    model = make_model(x_num_train.shape[1], prep.cat_cardinalities, bins).to(device)
    best_state, best_epoch, best_brier = train_epochs(
        model,
        (x_num_train, x_cat_train, y_train),
        device,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        validation_data=(x_num_val, x_cat_val, y_val),
    )
    print(f"best_epoch={best_epoch + 1} 2024_brier={best_brier:.6f}", flush=True)

    if args.skip_final_refit:
        model_state = best_state
        final_prep = prep
        final_bins = bins
        trained_rows = int(train_mask.sum())
    else:
        print(f"[2/2] final refit: 2019-2024 for {best_epoch + 1} epochs", flush=True)
        del model, x_num_train, x_cat_train, x_num_val, x_cat_val
        if device.type == "cuda":
            torch.cuda.empty_cache()
        set_seed(SEED)
        final_prep = Preprocessor.fit(frame, input_columns)
        x_num_all, x_cat_all = final_prep.transform(frame)
        y_all = frame[TARGET_COL].to_numpy(dtype=np.float32)
        final_bins = compute_numerical_bins(x_num_all, seed=SEED)
        model = make_model(x_num_all.shape[1], final_prep.cat_cardinalities, final_bins).to(device)
        model_state, _, _ = train_epochs(
            model,
            (x_num_all, x_cat_all, y_all),
            device,
            batch_size=args.batch_size,
            max_epochs=best_epoch + 1,
            patience=args.patience,
        )
        trained_rows = len(frame)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    final_prep.save(args.model_dir / "preprocessor.json")
    checkpoint = {
        "model_state": model_state,
        "bins": [value.detach().cpu() for value in final_bins],
        "model_config": dict(MODEL_CONFIG),
        "n_num_features": len(final_prep.numerical_columns),
        "cat_cardinalities": final_prep.cat_cardinalities,
        "validation_brier_2024": best_brier,
        "best_epoch": best_epoch + 1,
        "trained_rows": trained_rows,
        "seed": SEED,
    }
    torch.save(checkpoint, args.model_dir / "tabm.pt")
    report = {k: v for k, v in checkpoint.items() if k not in ("model_state", "bins")}
    (args.model_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {args.model_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
