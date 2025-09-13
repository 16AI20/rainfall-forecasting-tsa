"""
evaluation.py

This module provides utility functions to:
- Evaluate forecasting models using RMSE, R², and loss metrics.
- Plot training curves for diagnostics.
- Generate and save forecast visualizations for inspection.
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_scaler: Optional[object] = None
) -> Tuple[float, float, float]:
    """
    Evaluate model performance on a validation DataLoader.

    Args:
        model: Trained PyTorch model.
        val_loader: DataLoader containing validation data.
        criterion: Loss function (e.g., MSE).
        device: Device to perform evaluation on (cpu/cuda/mps).
        target_scaler: Optional scaler with inverse_transform() method.

    Returns:
        Tuple of (mean_val_loss, RMSE, R² score).
    """
    model.eval()
    val_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            val_loss += loss.item()

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)

    if target_scaler:
        y_true_np = target_scaler.inverse_transform(y_true_np)
        y_pred_np = target_scaler.inverse_transform(y_pred_np)

    rmse = float(np.sqrt(((y_pred_np - y_true_np) ** 2).mean()))
    r2 = float(r2_score(y_true_np, y_pred_np))

    logger.info("Evaluation completed - Loss: %.4f | RMSE: %.4f | R2: %.4f",
                val_loss / len(val_loader), rmse, r2)

    return val_loss / len(val_loader), rmse, r2


def plot_training_curves(
    train_loss: List[float],
    val_loss: List[float],
    train_rmse: List[float],
    val_rmse: List[float],
    train_r2: List[float],
    val_r2: List[float]
) -> None:
    """
    Plot training and validation metrics (Loss, RMSE, R²) over epochs.

    Args:
        train_loss: Training loss values.
        val_loss: Validation loss values.
        train_rmse: Training RMSE values.
        val_rmse: Validation RMSE values.
        train_r2: Training R² scores.
        val_r2: Validation R² scores.
    """
    logger.info("Plotting training curves...")
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_rmse, label='Train RMSE')
    plt.plot(epochs, val_rmse, label='Val RMSE')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("RMSE over Epochs")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_r2, label='Train R²')
    plt.plot(epochs, val_r2, label='Val R²')
    plt.xlabel("Epoch")
    plt.ylabel("R² Score")
    plt.title("R² over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()

