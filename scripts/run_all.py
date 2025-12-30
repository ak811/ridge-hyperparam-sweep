from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.datasets import load_diabetes_dataset, train_val_split_first90_last10
from src.features import (
    compute_scaling_params,
    scale_features,
    poly2_pairwise_products_plus_original,
)
from src.models import (
    fit_linear_regression,
    predict_linear_regression,
)
from src.metrics import mse
from src.experiments import ridge_sweep
from src.plotting import plot_ridge_mse


OUTPUTS_DIR = Path("outputs")
FIG_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"


def main() -> None:
    # ---- Load scaled dataset and split (like your notebook) ----
    X_scaled, y, feature_names = load_diabetes_dataset(scaled=True)
    split = train_val_split_first90_last10(X_scaled, y)

    # ---- Linear regression on original features ----
    theta_lin = fit_linear_regression(split.X_train, split.y_train, intercept=True)
    yhat_train = predict_linear_regression(theta_lin, split.X_train, intercept=True)
    yhat_val = predict_linear_regression(theta_lin, split.X_val, intercept=True)

    lin_train_mse = mse(split.y_train, yhat_train)
    lin_val_mse = mse(split.y_val, yhat_val)

    # ---- Polynomial features (pairwise products + originals) ----
    Xtr_poly = poly2_pairwise_products_plus_original(split.X_train)
    Xval_poly = poly2_pairwise_products_plus_original(split.X_val)

    theta_lin_poly = fit_linear_regression(Xtr_poly, split.y_train, intercept=True)
    yhat_train_poly = predict_linear_regression(theta_lin_poly, Xtr_poly, intercept=True)
    yhat_val_poly = predict_linear_regression(theta_lin_poly, Xval_poly, intercept=True)

    lin_poly_train_mse = mse(split.y_train, yhat_train_poly)
    lin_poly_val_mse = mse(split.y_val, yhat_val_poly)

    # ---- Ridge sweeps ----
    lambdas = np.logspace(-9, 0, num=50)

    ridge_orig = ridge_sweep(
        split.X_train, split.y_train, split.X_val, split.y_val, lambdas,
        intercept=True,
        regularize_intercept=False,
    )

    ridge_poly = ridge_sweep(
        Xtr_poly, split.y_train, Xval_poly, split.y_val, lambdas,
        intercept=True,
        regularize_intercept=False,
    )

    # ---- Save plots ----
    plot_ridge_mse(
        ridge_orig.lambdas, ridge_orig.mse_train, ridge_orig.mse_val,
        title="Ridge Regression: MSE vs. Lambda (Original Features)",
        save_path=FIG_DIR / "ridge_mse_original_features.png",
    )

    plot_ridge_mse(
        ridge_poly.lambdas, ridge_poly.mse_train, ridge_poly.mse_val,
        title="Ridge Regression: MSE vs. Lambda (Polynomial Features)",
        save_path=FIG_DIR / "ridge_mse_poly_features.png",
    )

    # ---- Patient prediction (your HW 3.9 flow) ----
    # Use unscaled features to compute scaling params that match sklearn's scaled dataset.
    X_unscaled, _, _ = load_diabetes_dataset(scaled=False)
    scaling_params = compute_scaling_params(X_unscaled)

    # Patient features in dataset order:
    # ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
    x_patient_unscaled = np.array(
        [60, 1, 26.5, 105, 199, 89.3, 55, 2.77, 4.8851, 79],
        dtype=float,
    )

    x_patient_scaled = scale_features(x_patient_unscaled, scaling_params)
    x_patient_scaled = x_patient_scaled.reshape(1, -1)

    x_patient_poly = poly2_pairwise_products_plus_original(x_patient_scaled)

    # Best model: ridge on polynomial features picked by lowest val MSE
    # Fit on polynomial TRAIN data (same as sweep)
    # (Using theta already returned by sweep is fine, but we recompute for clarity.)
    theta_best = ridge_poly.best_theta
    y_patient_pred = float(
        predict_linear_regression(theta_best, x_patient_poly, intercept=True)[0]
    )

    # ---- Print summary ----
    print("\n=== Dataset ===")
    print(f"Feature count: {split.X_train.shape[1]} (names: {feature_names})")
    print(f"Train size: {split.X_train.shape[0]} | Val size: {split.X_val.shape[0]}")

    print("\n=== Linear Regression (Original Features) ===")
    print(f"Train MSE: {lin_train_mse:.4f}")
    print(f"Val   MSE: {lin_val_mse:.4f}")

    print("\n=== Linear Regression (Polynomial Features) ===")
    print(f"Train MSE: {lin_poly_train_mse:.4f}")
    print(f"Val   MSE: {lin_poly_val_mse:.4f}")

    print("\n=== Ridge Sweep (Original Features) ===")
    print(f"Best lambda: {ridge_orig.best_lambda:.3e}")
    print(f"Best val MSE: {ridge_orig.best_val_mse:.4f}")

    print("\n=== Ridge Sweep (Polynomial Features) ===")
    print(f"Best lambda: {ridge_poly.best_lambda:.3e}")
    print(f"Best val MSE: {ridge_poly.best_val_mse:.4f}")

    print("\n=== Patient Prediction (Best Polynomial Ridge Model) ===")
    print(f"Predicted disease progression: {y_patient_pred:.4f}")

    print("\n=== Saved figures ===")
    print(f"- {FIG_DIR / 'ridge_mse_original_features.png'}")
    print(f"- {FIG_DIR / 'ridge_mse_poly_features.png'}")

    # ---- Optional JSON summary ----
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "linear_original": {"train_mse": lin_train_mse, "val_mse": lin_val_mse},
        "linear_poly": {"train_mse": lin_poly_train_mse, "val_mse": lin_poly_val_mse},
        "ridge_original": {
            "best_lambda": ridge_orig.best_lambda,
            "best_val_mse": ridge_orig.best_val_mse,
        },
        "ridge_poly": {
            "best_lambda": ridge_poly.best_lambda,
            "best_val_mse": ridge_poly.best_val_mse,
        },
        "patient_prediction": {
            "x_patient_unscaled": x_patient_unscaled.tolist(),
            "y_pred": y_patient_pred,
        },
    }

    with open(METRICS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote metrics: {METRICS_DIR / 'run_summary.json'}\n")


if __name__ == "__main__":
    main()
