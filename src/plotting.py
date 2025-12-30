from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_ridge_mse(
    lambdas: np.ndarray,
    mse_train: np.ndarray,
    mse_val: np.ndarray,
    *,
    title: str,
    save_path: str | Path,
) -> None:
    """
    Save a ridge MSE vs lambda plot with log-scaled x-axis.
    """
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    mse_train = np.asarray(mse_train, dtype=float).reshape(-1)
    mse_val = np.asarray(mse_val, dtype=float).reshape(-1)

    if not (lambdas.shape == mse_train.shape == mse_val.shape):
        raise ValueError("lambdas, mse_train, mse_val must have the same shape")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.semilogx(lambdas, mse_train, label="Training MSE", marker="o")
    plt.semilogx(lambdas, mse_val, label="Validation MSE", marker="o")
    plt.title(title)
    plt.xlabel("Lambda")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
