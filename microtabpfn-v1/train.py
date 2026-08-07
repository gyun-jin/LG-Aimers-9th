"""Pretrain microTabPFN and store real pitch data as inference context."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from microtabpfn import MicroTabPFN, MicroTabPFNClassifier, get_device, pretrain


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = ROOT.parent / "data" / "train.csv"
DEFAULT_MODEL_PATH = ROOT / "model" / "model.pkl"
TARGET_COLUMN = "control_success"
FEATURE_COLUMNS = [
    "balls_before",
    "strikes_before",
    "asof_pitcher_success_rate",
    "asof_batter_success_rate",
]
FILL_VALUES = {
    "balls_before": 0.0,
    "strikes_before": 0.0,
    "asof_pitcher_success_rate": 0.5,
    "asof_batter_success_rate": 0.5,
}
MODEL_CONFIG = {
    "n_features": len(FEATURE_COLUMNS),
    "embedding_size": 64,
    "n_heads": 4,
    "n_layers": 3,
}


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the four numeric inputs expected by microTabPFN."""
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"필수 피처 컬럼이 없습니다: {missing}")

    features = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.fillna(FILL_VALUES).astype("float32")


def sample_real_context(
    train_path: Path,
    context_per_class: int,
    validation_per_class: int,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly sample balanced context and validation rows from a large CSV."""
    keep_per_class = context_per_class + validation_per_class
    random_generator = np.random.default_rng(seed)
    kept: dict[int, pd.DataFrame] = {0: pd.DataFrame(), 1: pd.DataFrame()}
    use_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    for chunk_number, chunk in enumerate(
        pd.read_csv(train_path, usecols=use_columns, chunksize=chunk_size), start=1
    ):
        targets = pd.to_numeric(chunk[TARGET_COLUMN], errors="coerce")
        valid = targets.isin([0, 1])
        if not valid.any():
            continue

        sampled = make_features(chunk.loc[valid])
        sampled[TARGET_COLUMN] = targets.loc[valid].astype("int8").to_numpy()
        sampled["_sample_key"] = random_generator.random(len(sampled))

        for label in (0, 1):
            label_rows = sampled.loc[sampled[TARGET_COLUMN] == label]
            if label_rows.empty:
                continue
            pool = pd.concat([kept[label], label_rows], ignore_index=True)
            kept[label] = pool.nsmallest(keep_per_class, "_sample_key")

        if chunk_number % 10 == 0:
            print(f"데이터 청크 {chunk_number}개 처리")

    context_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    for label in (0, 1):
        rows = kept[label].sort_values("_sample_key").reset_index(drop=True)
        if len(rows) < keep_per_class:
            raise ValueError(
                f"클래스 {label} 데이터가 부족합니다: "
                f"필요 {keep_per_class}, 확보 {len(rows)}"
            )
        context_parts.append(rows.iloc[:context_per_class])
        validation_parts.append(rows.iloc[context_per_class:keep_per_class])

    context = pd.concat(context_parts, ignore_index=True).sample(
        frac=1.0, random_state=seed
    )
    validation = pd.concat(validation_parts, ignore_index=True).sample(
        frac=1.0, random_state=seed + 1
    )

    return (
        context[FEATURE_COLUMNS].to_numpy(dtype="float32"),
        context[TARGET_COLUMN].to_numpy(dtype="int64"),
        validation[FEATURE_COLUMNS].to_numpy(dtype="float32"),
        validation[TARGET_COLUMN].to_numpy(dtype="int64"),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--context-per-class", type=int, default=50)
    parser.add_argument("--validation-per-class", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    """Train, validate, and serialize one self-contained model artifact."""
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps는 1 이상이어야 합니다.")
    if args.context_per_class < 1 or args.validation_per_class < 1:
        raise ValueError("클래스별 context와 validation 크기는 1 이상이어야 합니다.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    print(f"실제 데이터 컨텍스트 추출: {args.train_data}")
    context_x, context_y, validation_x, validation_y = sample_real_context(
        args.train_data,
        args.context_per_class,
        args.validation_per_class,
        args.chunk_size,
        args.seed,
    )

    model = MicroTabPFN(**MODEL_CONFIG)
    model = pretrain(model, args.steps, args.learning_rate, device)
    classifier = MicroTabPFNClassifier(model, device).fit(context_x, context_y)
    validation_probability = np.concatenate(
        [
            classifier.predict_proba(validation_x[start : start + 256])[:, 1]
            for start in range(0, len(validation_x), 256)
        ]
    )
    validation_auc = roc_auc_score(validation_y, validation_probability)
    print(f"검증 ROC-AUC: {validation_auc:.6f}")

    artifact = {
        "format_version": 1,
        "source": "https://github.com/jxucoder/microTabPFN",
        "source_commit": "485d7247df6b3a7ed1f09a3dc351a224eef4bd1c",
        "feature_columns": FEATURE_COLUMNS,
        "fill_values": FILL_VALUES,
        "target_column": TARGET_COLUMN,
        "model_config": MODEL_CONFIG,
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "context_features": context_x,
        "context_labels": context_y,
        "seed": args.seed,
        "pretrain_steps": args.steps,
        "validation_rows": len(validation_y),
        "validation_auc": float(validation_auc),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as model_file:
        pickle.dump(artifact, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"모델 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
