from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScalingParams:
    mean: np.ndarray  # shape (d,)
    std: np.ndarray   # shape (d,)
    n: int            # number of samples used


def compute_scaling_params(X_unscaled: np.ndarray) -> ScalingParams:
    """
    Computes mean/std/n for the specific scaling used by sklearn diabetes 'scaled=True':
        x_scaled = (x - mean) / (std * sqrt(n))

    Args:
        X_unscaled: shape (n, d)

    Returns:
        ScalingParams
    """
    X_unscaled = np.asarray(X_unscaled, dtype=float)
    if X_unscaled.ndim != 2:
        raise ValueError(f"X_unscaled must be 2D, got shape {X_unscaled.shape}")

    mean = X_unscaled.mean(axis=0)
    std = X_unscaled.std(axis=0)

    # Avoid division by zero (shouldn't happen here, but humans love surprises)
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0

    return ScalingParams(mean=mean, std=std_safe, n=X_unscaled.shape[0])


def scale_features(X_unscaled: np.ndarray, params: ScalingParams) -> np.ndarray:
    """
    Apply diabetes scaling:
        x_scaled = (x - mean) / (std * sqrt(n))

    Works for a single vector or a matrix.

    Args:
        X_unscaled: shape (d,) or (n, d)
        params: ScalingParams computed from the full unscaled dataset

    Returns:
        scaled array with same shape as X_unscaled
    """
    X_unscaled = np.asarray(X_unscaled, dtype=float)
    denom = params.std * np.sqrt(params.n)
    return (X_unscaled - params.mean) / denom


def poly2_pairwise_products_plus_original(X: np.ndarray) -> np.ndarray:
    """
    Feature map matching your notebook's intent:
    - include ALL pairwise products x_i * x_j for i < j  (degree 2 cross terms)
    - append original features (degree 1 terms)

    Output dimension:
        D = d*(d-1)/2 + d

    Args:
        X: shape (n, d)

    Returns:
        X_poly: shape (n, D)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n, d = X.shape
    # number of i<j pairs
    num_pairs = d * (d - 1) // 2
    X_pairs = np.empty((n, num_pairs), dtype=float)

    col = 0
    for i in range(d):
        for j in range(i + 1, d):
            X_pairs[:, col] = X[:, i] * X[:, j]
            col += 1

    X_poly = np.concatenate([X_pairs, X], axis=1)
    return X_poly
