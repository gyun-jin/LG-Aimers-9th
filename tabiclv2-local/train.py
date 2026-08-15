"""Train a memory-conscious local TabICLv2 model and create predictions."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = ROOT.parent / "data" / "train.csv"
DEFAULT_TEST_PATH = ROOT.parent / "data" / "test.csv"
DEFAULT_SAMPLE_PATH = ROOT.parent / "data" / "sample_submission.csv"
DEFAULT_MODEL_PATH = ROOT / "model" / "classifier.pkl"
DEFAULT_METADATA_PATH = ROOT / "model" / "metadata.json"
DEFAULT_OUTPUT_PATH = ROOT / "output" / "submission.csv"
DEFAULT_OFFLOAD_DIR = ROOT / "offload"

ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"
CATEGORICAL_COLUMNS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

PRESETS = {
    "smoke": {"context_size": 1_000, "validation_size": 200},
    "safe": {"context_size": 10_000, "validation_size": 1_000},
    "full": {"context_size": 30_000, "validation_size": 2_000},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--prediction-output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--offload-dir", type=Path, default=DEFAULT_OFFLOAD_DIR)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--context-size", type=int, default=None)
    parser.add_argument("--validation-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Embed checkpoint weights in classifier.pkl (larger but self-contained).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, int]:
    context_size = (
        args.context_size
        if args.context_size is not None
        else PRESETS[args.preset]["context_size"]
    )
    validation_size = (
        args.validation_size
        if args.validation_size is not None
        else PRESETS[args.preset]["validation_size"]
    )
    for name, value in (
        ("context size", context_size),
        ("validation size", validation_size),
        ("chunk size", args.chunk_size),
        ("n estimators", args.n_estimators),
        ("batch size", args.batch_size),
        ("n jobs", args.n_jobs),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    if not args.train_data.is_file():
        raise FileNotFoundError(args.train_data)
    if not args.test_data.is_file():
        raise FileNotFoundError(args.test_data)
    return context_size, validation_size


def get_feature_columns(train_path: Path) -> list[str]:
    columns = pd.read_csv(train_path, nrows=0).columns.tolist()
    missing = {ID_COLUMN, TARGET_COLUMN} - set(columns)
    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")
    return [column for column in columns if column not in {ID_COLUMN, TARGET_COLUMN}]


def sample_csv_uniformly(
    path: Path,
    feature_columns: list[str],
    sample_size: int,
    chunk_size: int,
    seed: int,
) -> pd.DataFrame:
    """Keep the rows with the globally smallest random keys.

    This is an exact uniform sample without loading the 1.47M-row CSV at once.
    """
    rng = np.random.default_rng(seed)
    reservoir = pd.DataFrame()
    use_columns = feature_columns + [TARGET_COLUMN]
    rows_seen = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, usecols=use_columns, chunksize=chunk_size), start=1
    ):
        target = pd.to_numeric(chunk[TARGET_COLUMN], errors="coerce")
        chunk = chunk.loc[target.isin([0, 1])].copy()
        chunk[TARGET_COLUMN] = target.loc[chunk.index].astype("int8")
        chunk["_sample_key"] = rng.random(len(chunk))
        rows_seen += len(chunk)

        reservoir = pd.concat([reservoir, chunk], ignore_index=True)
        if len(reservoir) > sample_size:
            reservoir = reservoir.nsmallest(sample_size, "_sample_key")

        print(
            f"  chunk {chunk_number:02d}: valid rows={rows_seen:,}, "
            f"reservoir={len(reservoir):,}",
            flush=True,
        )

    if len(reservoir) < sample_size:
        raise ValueError(
            f"Requested {sample_size:,} rows but only found {len(reservoir):,} valid rows"
        )
    return reservoir.drop(columns="_sample_key").reset_index(drop=True)


def prepare_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing feature columns: {missing}")

    features = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        if column in CATEGORICAL_COLUMNS:
            # Strings force IDs and other codes to be treated as categories rather
            # than continuous quantities. Missing values receive their own category.
            features[column] = features[column].astype("string").fillna("__MISSING__")
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce").astype(
                "float32"
            )
    return features


def predict_class_one(model: TabICLClassifier, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features))
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Expected binary classes containing 1, got {classes.tolist()}")
    return probabilities[:, int(matches[0])]


def build_submission(
    test_frame: pd.DataFrame,
    probabilities: np.ndarray,
    sample_path: Path,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            ID_COLUMN: test_frame[ID_COLUMN].astype(str),
            TARGET_COLUMN: np.clip(probabilities, 0.0, 1.0),
        }
    )
    if not sample_path.is_file():
        return predictions

    sample_ids = pd.read_csv(sample_path, usecols=[ID_COLUMN])
    sample_ids[ID_COLUMN] = sample_ids[ID_COLUMN].astype(str)
    output = sample_ids.merge(predictions, on=ID_COLUMN, how="left", validate="one_to_one")
    if output[TARGET_COLUMN].isna().any():
        raise ValueError("sample_submission.csv and test.csv row_id values do not match")
    return output


def main() -> None:
    args = parse_args()
    context_size, validation_size = validate_args(args)
    feature_columns = get_feature_columns(args.train_data)
    total_sample_size = context_size + validation_size
    started = time.perf_counter()

    print(
        f"[1/5] Uniformly sampling {total_sample_size:,} rows from {args.train_data}",
        flush=True,
    )
    sampled = sample_csv_uniformly(
        args.train_data,
        feature_columns,
        total_sample_size,
        args.chunk_size,
        args.seed,
    )
    context, validation = train_test_split(
        sampled,
        train_size=context_size,
        test_size=validation_size,
        stratify=sampled[TARGET_COLUMN],
        random_state=args.seed,
    )
    x_context = prepare_features(context, feature_columns)
    y_context = context[TARGET_COLUMN].to_numpy(dtype="int64")
    x_validation = prepare_features(validation, feature_columns)
    y_validation = validation[TARGET_COLUMN].to_numpy(dtype="int64")
    print(
        f"  context positive rate={y_context.mean():.4f}; "
        f"validation positive rate={y_validation.mean():.4f}",
        flush=True,
    )

    args.offload_dir.mkdir(parents=True, exist_ok=True)
    print("[2/5] Initializing TabICLv2 on CPU", flush=True)
    model = TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        kv_cache=False,
        device="cpu",
        use_amp=False,
        use_fa3=False,
        offload_mode="disk",
        disk_offload_dir=str(args.offload_dir),
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbose=True,
    )
    model.fit(x_context, y_context)

    print(f"[3/5] Validating on {validation_size:,} held-out rows", flush=True)
    validation_probability = predict_class_one(model, x_validation)
    validation_auc = roc_auc_score(y_validation, validation_probability)
    validation_log_loss = log_loss(y_validation, validation_probability, labels=[0, 1])
    print(f"  ROC-AUC={validation_auc:.6f}; log loss={validation_log_loss:.6f}", flush=True)

    print(f"[4/5] Saving fitted model to {args.model_output}", flush=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(
        str(args.model_output),
        save_model_weights=args.save_weights,
        save_training_data=True,
        save_kv_cache=False,
    )
    metadata = {
        "format_version": 1,
        "feature_columns": feature_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "target_column": TARGET_COLUMN,
        "context_size": context_size,
        "validation_size": validation_size,
        "n_estimators": args.n_estimators,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "validation_auc": float(validation_auc),
        "validation_log_loss": float(validation_log_loss),
        "weights_embedded": bool(args.save_weights),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[5/5] Predicting {args.test_data}", flush=True)
    test_frame = pd.read_csv(args.test_data)
    test_features = prepare_features(test_frame, feature_columns)
    test_probability = predict_class_one(model, test_features)
    submission = build_submission(test_frame, test_probability, args.sample_submission)
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.prediction_output, index=False)
    print(
        f"Done in {(time.perf_counter() - started) / 60:.1f} min: "
        f"{args.prediction_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
