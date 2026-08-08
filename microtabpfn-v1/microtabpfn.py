"""Minimal TabPFN implementation adapted from jxucoder/microTabPFN.

The upstream project is distributed under the MIT License. Its license text
is included in LICENSE.microTabPFN next to this file.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(requested: str = "auto") -> torch.device:
    """Return the requested accelerator, or the best available device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_scm(n_samples: int, n_features: int, device: torch.device) -> torch.Tensor:
    """Generate synthetic features with causal dependencies."""
    features = torch.zeros(n_samples, n_features, device=device)
    features[:, 0] = torch.randn(n_samples, device=device)
    for index in range(1, n_features):
        weights = torch.randn(index, device=device)
        signal = features[:, :index] @ weights / index**0.5
        features[:, index] = signal + torch.randn(n_samples, device=device) * 0.5
    return features


def sample_bnn(features: torch.Tensor) -> torch.Tensor:
    """Generate binary labels with a random two-layer neural network."""
    n_features = features.shape[1]
    hidden_size = 16
    device = features.device
    weight_1 = torch.randn(n_features, hidden_size, device=device) / n_features**0.5
    bias_1 = torch.randn(hidden_size, device=device) * 0.5
    weight_2 = torch.randn(hidden_size, 1, device=device) / hidden_size**0.5
    bias_2 = torch.randn(1, device=device) * 0.5
    logits = torch.tanh(features @ weight_1 + bias_1) @ weight_2 + bias_2
    return (logits > 0).long().squeeze(-1)


def sample_task(
    n_train: int,
    n_test: int,
    n_features: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one synthetic in-context classification task."""
    features = sample_scm(n_train + n_test, n_features, device)
    labels = sample_bnn(features)
    return (
        features[:n_train],
        labels[:n_train],
        features[n_train:],
        labels[n_train:],
    )


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


def pretrain(
    model: MicroTabPFN,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> MicroTabPFN:
    """Train the model on freshly generated synthetic classification tasks."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    print(f"Pretraining on {device} for {steps} steps")
    for step in range(steps):
        n_train = random.randint(30, 100)
        n_test = random.randint(10, 50)
        x_train, y_train, x_test, y_test = sample_task(
            n_train, n_test, model.n_features, device
        )
        loss = F.cross_entropy(model(x_train, y_train, x_test), y_test)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:5d} | loss {loss.item():.4f}")
    return model


class MicroTabPFNClassifier:
    """Small sklearn-style wrapper that stores real data as inference context."""

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

    def predict_proba(self, features) -> torch.Tensor:
        test = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        test = (test - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(self.features, self.labels, test)
            return F.softmax(logits, dim=-1).cpu().numpy()
