"""Load a trained microTabPFN artifact and create control predictions.

The microTabPFN model code in this file is adapted from
https://github.com/jxucoder/microTabPFN (MIT License).

MIT License

Copyright (c) 2025 J Xu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "model" / "model.pkl"
DEFAULT_TEST_PATH = ROOT.parent / "data" / "test.csv"
DEFAULT_SAMPLE_PATH = ROOT.parent / "data" / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = ROOT / "prediction.csv"
ID_COLUMN = "row_id"


def get_device(requested: str = "auto") -> torch.device:
    """Return the requested accelerator, or the best available device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Block(nn.Module):
    """Apply attention over both feature columns and sample rows."""

    def __init__(self, embedding_size: int, n_heads: int) -> None:
        super().__init__()
        self.column_attention = nn.MultiheadAttention(
            embedding_size, n_heads, batch_first=True
        )
        self.row_attention = nn.MultiheadAttention(
            embedding_size, n_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 2),
            nn.GELU(),
            nn.Linear(embedding_size * 2, embedding_size),
        )
        self.norm_1 = nn.LayerNorm(embedding_size)
        self.norm_2 = nn.LayerNorm(embedding_size)
        self.norm_3 = nn.LayerNorm(embedding_size)

    def forward(self, inputs: torch.Tensor, n_train: int) -> torch.Tensor:
        attended, _ = self.column_attention(inputs, inputs, inputs)
        inputs = self.norm_1(inputs + attended)

        transposed = inputs.transpose(0, 1)
        train_rows = transposed[:, :n_train]
        test_rows = transposed[:, n_train:]
        train_attended, _ = self.row_attention(train_rows, train_rows, train_rows)
        test_attended, _ = self.row_attention(test_rows, train_rows, train_rows)
        combined = torch.cat(
            [train_rows + train_attended, test_rows + test_attended], dim=1
        )
        inputs = self.norm_2(combined.transpose(0, 1))
        return self.norm_3(inputs + self.feed_forward(inputs))


class MicroTabPFN(nn.Module):
    """Small in-context transformer for four-feature binary classification."""

    def __init__(
        self,
        n_features: int = 4,
        embedding_size: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.feature_embedding = nn.Linear(1, embedding_size)
        self.label_embedding = nn.Linear(1, embedding_size)
        self.blocks = nn.ModuleList(
            [Block(embedding_size, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, 2),
        )

    def forward(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
    ) -> torch.Tensor:
        n_train = len(train_features)
        n_test = len(test_features)
        all_features = torch.cat([train_features, test_features])
        embedded_features = self.feature_embedding(all_features.unsqueeze(-1))
        label_values = torch.cat(
            [
                train_labels.float(),
                torch.full((n_test,), 0.5, device=train_features.device),
            ]
        )
        embedded_labels = self.label_embedding(label_values.unsqueeze(-1)).unsqueeze(1)
        hidden = torch.cat([embedded_features, embedded_labels], dim=1)
        for block in self.blocks:
            hidden = block(hidden, n_train)
        return self.head(hidden[n_train:, -1])


class MicroTabPFNClassifier:
    """Small wrapper that stores real pitch data as inference context."""

    def __init__(self, model: MicroTabPFN, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    def fit(self, features, labels) -> "MicroTabPFNClassifier":
        context = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        self.labels = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        self.mean = context.mean(dim=0)
        self.std = context.std(dim=0) + 1e-8
        self.features = (context - self.mean) / self.std
        return self

    def predict_proba(self, features) -> np.ndarray:
        test = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        test = (test - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(self.features, self.labels, test)
            return F.softmax(logits, dim=-1).cpu().numpy()


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
