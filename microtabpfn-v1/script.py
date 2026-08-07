"""Load a trained microTabPFN artifact and create control predictions."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from microtabpfn import MicroTabPFN, MicroTabPFNClassifier, get_device


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "model" / "model.pkl"
DEFAULT_TEST_PATH = ROOT.parent / "data" / "test.csv"
DEFAULT_SAMPLE_PATH = ROOT.parent / "data" / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = ROOT / "prediction.csv"
ID_COLUMN = "row_id"


def make_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> np.ndarray:
    """Reproduce the numeric feature preparation used during training."""
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"필수 피처 컬럼이 없습니다: {missing}")

    features = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features = features.fillna(fill_values).astype("float32")
    return features.to_numpy(dtype="float32")


def load_artifact(path: Path) -> dict:
    """Load and minimally validate a trusted local model artifact."""
    with path.open("rb") as model_file:
        artifact = pickle.load(model_file)
    required = {
        "format_version",
        "feature_columns",
        "fill_values",
        "target_column",
        "model_config",
        "state_dict",
        "context_features",
        "context_labels",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"모델 파일에 필요한 항목이 없습니다: {missing}")
    if artifact["format_version"] != 1:
        raise ValueError(f"지원하지 않는 모델 형식: {artifact['format_version']}")
    return artifact


def predict_in_batches(
    classifier: MicroTabPFNClassifier,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict class-1 probabilities without building a large attention matrix."""
    probabilities: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        probabilities.append(classifier.predict_proba(batch)[:, 1])
    if not probabilities:
        return np.empty(0, dtype="float32")
    return np.concatenate(probabilities)


def build_output(
    test_frame: pd.DataFrame,
    probabilities: np.ndarray,
    target_column: str,
    sample_submission_path: Path,
) -> pd.DataFrame:
    """Match predictions to sample_submission.csv by row_id."""
    if ID_COLUMN not in test_frame.columns:
        raise ValueError(f"테스트 데이터에 {ID_COLUMN} 컬럼이 없습니다.")
    predictions = pd.DataFrame(
        {ID_COLUMN: test_frame[ID_COLUMN].astype(str), target_column: probabilities}
    )

    if not sample_submission_path.exists():
        return predictions

    sample = pd.read_csv(sample_submission_path, usecols=[ID_COLUMN])
    output = sample.merge(predictions, on=ID_COLUMN, how="left", validate="one_to_one")
    if output[target_column].isna().any():
        raise ValueError("sample_submission과 test의 row_id가 일치하지 않습니다.")
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    """Load the model, predict the test rows, and write a submission-shaped CSV."""
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    artifact = load_artifact(args.model)
    test_frame = pd.read_csv(args.input)
    test_features = make_features(
        test_frame,
        artifact["feature_columns"],
        artifact["fill_values"],
    )

    device = get_device(args.device)
    model = MicroTabPFN(**artifact["model_config"])
    model.load_state_dict(artifact["state_dict"])
    classifier = MicroTabPFNClassifier(model, device).fit(
        artifact["context_features"], artifact["context_labels"]
    )
    probabilities = predict_in_batches(classifier, test_features, args.batch_size)
    output = build_output(
        test_frame,
        probabilities,
        artifact["target_column"],
        args.sample_submission,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"예측 완료: {args.output} ({len(output)}행, device={device})")


if __name__ == "__main__":
    main()
