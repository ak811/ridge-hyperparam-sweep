from __future__ import annotations

import numpy as np


def _add_intercept_column(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    return np.c_[np.ones((X.shape[0], 1), dtype=float), X]


def fit_linear_regression(X: np.ndarray, y: np.ndarray, *, intercept: bool = True) -> np.ndarray:
    """
    Closed-form linear regression (least squares).
    Uses pseudo-inverse for numerical stability.

    Model:
        y_hat = theta0 + x^T theta   if intercept=True
        y_hat = x^T theta            if intercept=False
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if intercept:
        X_design = _add_intercept_column(X)
    else:
        X_design = X

    # theta = (X^T X)^(-1) X^T y  -> use pinv for stability
    theta = np.linalg.pinv(X_design) @ y
    return theta


def predict_linear_regression(theta: np.ndarray, X: np.ndarray, *, intercept: bool = True) -> np.ndarray:
    """
    Predict using theta from fit_linear_regression / fit_ridge_regression.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)

    if intercept:
        X_design = _add_intercept_column(X)
    else:
        X_design = X

    return X_design @ theta


def fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    lmbda: float,
    *,
    intercept: bool = True,
    regularize_intercept: bool = False,
) -> np.ndarray:
    """
    Closed-form ridge regression.

    Objective:
        (1/n) ||X theta - y||^2 + lmbda * ||theta||^2

    Implementation detail:
    - If intercept=True, we add an intercept column.
    - By default, we do NOT regularize the intercept term (common convention).
      Your notebook regularized everything. This version is more standard but configurable.

    Args:
        X: (n, d)
        y: (n,)
        lmbda: regularization strength (>0 recommended)
        intercept: whether to include intercept
        regularize_intercept: if True, penalize theta0 too

    Returns:
        theta vector (d+1,) if intercept else (d,)
    """
    if lmbda < 0:
        raise ValueError("lmbda must be >= 0")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    X_design = _add_intercept_column(X) if intercept else X
    n, m = X_design.shape

    I = np.eye(m, dtype=float)
    if intercept and not regularize_intercept:
        I[0, 0] = 0.0

    # Match your notebook scaling style: lmbda * n * I
    # (equivalent to lmbda in many derivations depending on the 1/n placement)
    A = (X_design.T @ X_design) + (lmbda * n) * I
    b = X_design.T @ y

    theta = np.linalg.solve(A, b)
    return theta
