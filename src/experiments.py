from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import fit_ridge_regression, predict_linear_regression
from .metrics import mse


@dataclass(frozen=True)
class RidgeSweepResult:
    lambdas: np.ndarray
    mse_train: np.ndarray
    mse_val: np.ndarray
    best_lambda: float
    best_theta: np.ndarray
    best_val_mse: float


def ridge_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: np.ndarray,
    *,
    intercept: bool = True,
    regularize_intercept: bool = False,
) -> RidgeSweepResult:
    """
    Train ridge regression models for each lambda and track train/val MSE.
    Picks the model with the lowest validation MSE.

    Returns:
        RidgeSweepResult
    """
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    if lambdas.size == 0:
        raise ValueError("lambdas must be non-empty")

    train_mses = np.empty_like(lambdas)
    val_mses = np.empty_like(lambdas)
    thetas: list[np.ndarray] = []

    for idx, lmbda in enumerate(lambdas):
        theta = fit_ridge_regression(
            X_train, y_train, float(lmbda),
            intercept=intercept,
            regularize_intercept=regularize_intercept,
        )
        thetas.append(theta)

        yhat_train = predict_linear_regression(theta, X_train, intercept=intercept)
        yhat_val = predict_linear_regression(theta, X_val, intercept=intercept)

        train_mses[idx] = mse(y_train, yhat_train)
        val_mses[idx] = mse(y_val, yhat_val)

    best_idx = int(np.argmin(val_mses))
    return RidgeSweepResult(
        lambdas=lambdas,
        mse_train=train_mses,
        mse_val=val_mses,
        best_lambda=float(lambdas[best_idx]),
        best_theta=thetas[best_idx],
        best_val_mse=float(val_mses[best_idx]),
    )
