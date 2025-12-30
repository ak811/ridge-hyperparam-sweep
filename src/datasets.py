from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn import datasets


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


def load_diabetes_dataset(*, scaled: bool = True) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load the sklearn Diabetes dataset.

    Args:
        scaled: If True, uses sklearn's default standardized features.
                If False, loads original (unscaled) features.

    Returns:
        (X, y, feature_names)
    """
    ds = datasets.load_diabetes(scaled=scaled)
    X = np.asarray(ds.data, dtype=float)
    y = np.asarray(ds.target, dtype=float)
    feature_names = list(ds.feature_names)
    return X, y, feature_names


def train_val_split_first90_last10(X: np.ndarray, y: np.ndarray) -> DatasetSplit:
    """
    Split by order: first 90% train, last 10% validation (exactly like your notebook).

    Args:
        X: shape (n, d)
        y: shape (n,)

    Returns:
        DatasetSplit
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of rows")

    n = X.shape[0]
    train_len = int(n * 0.9)
    return DatasetSplit(
        X_train=X[:train_len].copy(),
        y_train=y[:train_len].copy(),
        X_val=X[train_len:].copy(),
        y_val=y[train_len:].copy(),
    )
