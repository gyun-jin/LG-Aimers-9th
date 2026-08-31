from __future__ import annotations

import warnings

import numpy as np
import torch
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from tabm import TabM


MODEL_CONFIG = {
    "arch_type": "tabm",
    "k": 16,
    "n_blocks": 3,
    "d_block": 256,
    "dropout": 0.15,
    "d_embedding": 16,
    "n_bins": 32,
}


def compute_numerical_bins(
    x_num: np.ndarray,
    *,
    n_bins: int = MODEL_CONFIG["n_bins"],
    max_rows: int = 200_000,
    seed: int = 42,
) -> list[torch.Tensor]:
    if len(x_num) > max_rows:
        idx = np.random.default_rng(seed).choice(len(x_num), size=max_rows, replace=False)
        x_num = x_num[idx]
    x_num = np.ascontiguousarray(x_num.copy())
    # compute_bins requires at least two distinct values in every column. A feature
    # can become constant in a temporal split, so give such columns a harmless tiny
    # interval while leaving every real observation at the interval centre.
    column_min = x_num.min(axis=0)
    column_max = x_num.max(axis=0)
    constant = np.flatnonzero(column_min == column_max)
    if len(constant):
        if len(x_num) < 2:
            raise ValueError("At least two rows are required to compute numerical bins")
        x_num[0, constant] = column_min[constant] - 1e-4
        x_num[1, constant] = column_max[constant] + 1e-4
    return compute_bins(torch.from_numpy(x_num), n_bins=n_bins)


def make_model(
    n_num_features: int,
    cat_cardinalities: list[int],
    bins: list[torch.Tensor],
    config: dict | None = None,
) -> TabM:
    cfg = dict(MODEL_CONFIG if config is None else config)
    with warnings.catch_warnings():
        # Binary/indicator features naturally have one interval; this is valid and
        # equivalent to min-max scaling, so the package warning is only noise here.
        warnings.filterwarnings(
            "ignore", message=r"The \d+-th feature has just two bin edges.*"
        )
        embeddings = PiecewiseLinearEmbeddings(
            bins,
            d_embedding=int(cfg["d_embedding"]),
            activation=False,
            version="B",
        )
    return TabM.make(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        num_embeddings=embeddings,
        arch_type=cfg["arch_type"],
        k=int(cfg["k"]),
        n_blocks=int(cfg["n_blocks"]),
        d_block=int(cfg["d_block"]),
        dropout=float(cfg["dropout"]),
    )
