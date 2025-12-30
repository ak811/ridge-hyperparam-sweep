from __future__ import annotations

import numpy as np

from src.datasets import load_diabetes_dataset, train_val_split_first90_last10
from src.features import (
    compute_scaling_params,
    scale_features,
    poly2_pairwise_products_plus_original,
)
from src.experiments import ridge_sweep
from src.models import predict_linear_regression


def main() -> None:
    # Load scaled dataset for training/splitting
    X_scaled, y, feature_names = load_diabetes_dataset(scaled=True)
    split = train_val_split_first90_last10(X_scaled, y)

    # Polynomial features
    Xtr_poly = poly2_pairwise_products_plus_original(split.X_train)
    Xval_poly = poly2_pairwise_products_plus_original(split.X_val)

    # Ridge sweep to find best lambda on poly features
    lambdas = np.logspace(-9, 0, num=50)
    ridge_poly = ridge_sweep(
        Xtr_poly, split.y_train, Xval_poly, split.y_val, lambdas,
        intercept=True,
        regularize_intercept=False,
    )

    # Compute scaling params from unscaled dataset
    X_unscaled, _, _ = load_diabetes_dataset(scaled=False)
    params = compute_scaling_params(X_unscaled)

    # Patient feature vector in dataset order:
    # ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
    x_patient_unscaled = np.array(
        [60, 1, 26.5, 105, 199, 89.3, 55, 2.77, 4.8851, 79],
        dtype=float,
    )
    x_patient_scaled = scale_features(x_patient_unscaled, params).reshape(1, -1)
    x_patient_poly = poly2_pairwise_products_plus_original(x_patient_scaled)

    y_pred = float(predict_linear_regression(ridge_poly.best_theta, x_patient_poly, intercept=True)[0])

    print("=== Best polynomial ridge model ===")
    print(f"Best lambda: {ridge_poly.best_lambda:.3e}")
    print(f"Validation MSE: {ridge_poly.best_val_mse:.4f}")
    print("\n=== Patient prediction ===")
    print(f"Predicted disease progression: {y_pred:.4f}")


if __name__ == "__main__":
    main()
