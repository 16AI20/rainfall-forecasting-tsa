#!/usr/bin/env python
"""
train.py

Time-series forecasting training script refactored from the original image-classification version.
It retains robust logging and MLflow integration, while adapting the training loop for
continuous targets using RMSE and R² as evaluation metrics.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import evaluate, plot_training_curves
from .general_utils import mlflow_init, mlflow_log, mlflow_pytorch_call, negative_log_likelihood
from .model_factory import create_model
from .time_series_dataset import TimeSeriesDataset

# Initialize module-level logger
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# MLflow constants
# -----------------------------------------------------------------------------
MLFLOW_LOG_PARAM = "log_param"
MLFLOW_LOG_PARAMS = "log_params"
MLFLOW_LOG_METRIC = "log_metric"
MLFLOW_LOG_METRICS = "log_metrics"
MLFLOW_LOG_ARTIFACT = "log_artifact"
MLFLOW_LOG_ARTIFACTS = "log_artifacts"
MLFLOW_LOG_DICT = "log_dict"
MLFLOW_LOG_FIGURE = "log_figure"
MLFLOW_LOG_TABLE = "log_table"
MLFLOW_LOG_IMAGE = "log_image"
MLFLOW_LOG_INPUT = "log_input"
MLFLOW_LOG_INPUTS = "log_inputs"
MLFLOW_LOG_MODEL = "log_model"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    feature_scaler: Optional[Any] = None,
    target_scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    probabilistic: bool = False
) -> None:
    """
    Trains a time-series forecasting model with validation, logging, and checkpointing.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (e.g. MSELoss).
        optimizer (optim.Optimizer): Optimizer instance.
        device (torch.device): Device to train on (CPU, CUDA, or MPS).
        cfg (DictConfig): Hydra configuration object.
        feature_scaler (Optional[Any], optional): Scaler used on input features. Defaults to None.
        target_scaler (Optional[Any], optional): Scaler used on target values. Defaults to None.
        scheduler (Optional[Any], optional): Learning rate scheduler. Defaults to None.
        probabilistic (bool, optional): If True, uses probabilistic model with NLL loss. Defaults to False.
    """
    _validate_inputs(model, train_loader, val_loader)
    _log_training_start(device)
    
    # Extract config values
    seed, exp_name, resume, run_name, ckpt_path, artifact_dir, epochs, patience = _extract_cfg(cfg)

    # Initialize metric histories
    train_hist, val_hist = _init_metric_histories()
    best_val_loss: float = float("inf")
    epochs_no_improve: int = 0
    start_epoch: int = 0

    # MLflow run setup
    mlflow_status, mlflow_run, step_offset = _init_mlflow(exp_name, run_name, cfg, resume)
    _resume_checkpoint_if_needed(resume, ckpt_path, device, model, optimizer, train_hist, val_hist)
    _log_static_hyperparams(mlflow_status, train_loader, val_loader, optimizer, seed, epochs, cfg)

    # -----------------------------
    # Main training loop
    # -----------------------------
    for epoch in range(start_epoch, epochs):
        mlflow_step = epoch + step_offset

        train_metrics, val_metrics = _train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            probabilistic=probabilistic,
            target_scaler=target_scaler
        )

        _update_histories(train_hist, val_hist, train_metrics, val_metrics)

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        # Save best model and log metrics
        if val_metrics["loss"] < best_val_loss:
            _save_best_model_and_log_metrics(
                model=model,
                optimizer=optimizer,
                checkpoint_path=ckpt_path,
                mlflow_step=mlflow_step,
                train_hist=train_hist,
                val_hist=val_hist,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                mlflow_status=mlflow_status,
                log_model=True
            )
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
        else:
            _save_best_model_and_log_metrics(
                model=model,
                optimizer=optimizer,
                checkpoint_path=ckpt_path,
                mlflow_step=mlflow_step,
                train_hist=train_hist,
                val_hist=val_hist,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                mlflow_status=mlflow_status,
                log_model=False
            )
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                logger.warning("Early stopping triggered.")
                break

    logger.info(f"Final training history length: {len(train_hist['loss'])}")
    
    _finalize_training(
        train_hist=train_hist,
        val_hist=val_hist,
        mlflow_status=mlflow_status,
        artifact_dir=artifact_dir,
        model=model,
        model_name=cfg.model_cfg.registered_model_name,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler
    )

def _validate_inputs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> None:
    """
    Performs basic sanity checks on the model and data loaders.

    Verifies that:
    - The model is an instance of `torch.nn.Module`.
    - The inputs `x` are 3-dimensional (B, seq_len, n_features).
    - The targets `y` are 2-dimensional (B, pred_len).
    - No NaNs are present in inputs or targets.

    Raises:
        TypeError: If model is not a PyTorch module.
        ValueError: If dimensions are incorrect or data contains NaNs.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Model must be a `torch.nn.Module`.")

    for name, loader in {"train_loader": train_loader, "val_loader": val_loader}.items():
        try:
            x, y = next(iter(loader))
        except StopIteration:
            raise ValueError(f"{name} is empty.")

        if x.dim() != 3:
            raise ValueError(f"Expected x to be 3-D (B, seq_len, n_feat); got {x.shape} from {name}.")
        if y.dim() != 2:
            raise ValueError(f"Expected y to be 2-D (B, pred_len); got {y.shape} from {name}.")
        if torch.isnan(x).any() or torch.isnan(y).any():
            raise ValueError(f"{name} contains NaN values.")

    logger.info("Input sanity checks passed.")


def _extract_cfg(cfg: DictConfig) -> Tuple[int, str, bool, str, str, str, int, int]:
    """
    Extracts core training configuration values from a Hydra DictConfig.

    Args:
        cfg (DictConfig): Configuration object containing model and logging parameters.

    Returns:
        Tuple[int, str, bool, str, str, str, int, int]: A tuple containing:
            - seed (int): Random seed for reproducibility.
            - exp_name (str): Experiment name.
            - resume (bool): Whether to resume from checkpoint.
            - run_name (str): Model/run name.
            - checkpoint_path (str): Path to checkpoint file.
            - artifact_dir (str): Path to store MLflow artifacts.
            - epochs (int): Number of training epochs.
            - patience (int): Early stopping patience.
    """
    seed: int = cfg.seed
    exp_name: str = cfg.exp_name
    resume: bool = cfg.resume
    run_name: str = cfg.model_cfg.registered_model_name
    checkpoint_path: str = os.path.join(cfg.checkpoint.dir, run_name + ".pth")
    artifact_dir: str = cfg.logging.artifact_dir
    epochs: int = cfg.model_cfg.epochs
    patience: int = cfg.model_cfg.patience

    return seed, exp_name, resume, run_name, checkpoint_path, artifact_dir, epochs, patience


def _init_metric_histories() -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Initializes metric history dictionaries for tracking training and validation performance.

    Returns:
        Tuple[Dict[str, List[float]], Dict[str, List[float]]]: Empty training and validation histories.
    """
    train_metrics: Dict[str, List[float]] = {"loss": [], "rmse": [], "r2": []}
    val_metrics: Dict[str, List[float]] = {"loss": [], "rmse": [], "r2": []}
    return train_metrics, val_metrics


def _log_training_start(device: torch.device) -> None:
    """
    Logs a message indicating the start of training on the specified device.

    Args:
        device (torch.device): The device being used for training.
    """
    logger.info(f"Starting training on device: {device}")


def _init_mlflow(
    exp_name: str,
    run_name: str,
    cfg: DictConfig,
    resume: bool
) -> Tuple[bool, Any, int]:
    """
    Initializes MLflow experiment tracking.

    Args:
        exp_name (str): Name of the MLflow experiment.
        run_name (str): Name of the run (used as identifier in UI).
        cfg (DictConfig): Configuration containing MLflow settings.
        resume (bool): Whether to resume a previous run.

    Returns:
        Tuple[bool, Any, int]: 
            - success status (bool),
            - MLflow run object,
            - step offset (for logging continuation).
    """
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI")

    return mlflow_init(
        tracking_uri=tracking_uri,
        exp_name=exp_name,
        run_name=run_name,
        setup_mlflow=cfg.mlflow.setup,
        autolog=cfg.mlflow.autolog,
        resume=resume,
    )

def _resume_checkpoint_if_needed(
    resume: bool,
    checkpoint_path: str,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]]
) -> None:
    """
    Loads model and optimizer states from a checkpoint file, if applicable.

    Args:
        resume (bool): Whether to resume training from a checkpoint.
        checkpoint_path (str): Path to the checkpoint file (.pth).
        device (torch.device): Device to map tensors to (e.g., "cpu", "cuda", "mps").
        model (nn.Module): Model instance to load weights into.
        optimizer (optim.Optimizer): Optimizer instance to restore state.
        train_hist (Dict[str, List[float]]): Training metric history to restore.
        val_hist (Dict[str, List[float]]): Validation metric history to restore.
    """
    if resume and os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimiser_state_dict"])

        for key in train_hist:
            train_hist[key] = checkpoint.get(f"train_{key}_history", [])
            val_hist[key] = checkpoint.get(f"val_{key}_history", [])
        logger.info("Checkpoint restored successfully.")
    else:
        logger.warning("No checkpoint found or resume disabled. Starting from scratch.")


def _log_static_hyperparams(
    status: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    seed: int,
    epochs: int,
    cfg: DictConfig
) -> None:
    """
    Logs static hyperparameters to MLflow (if enabled).

    Args:
        status (bool): True if MLflow is active.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (optim.Optimizer): Optimizer for the model.
        seed (int): Random seed used in training.
        epochs (int): Total number of training epochs.
        cfg (DictConfig): Model configuration object.
    """
    params: Dict[str, Any] = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "train_batch_size": train_loader.batch_size,
        "val_batch_size": val_loader.batch_size,
        "seed": seed,
        "epochs": epochs,
        "model": cfg.model_cfg.model_type,
    }

    logger.info("Logging static hyperparameters to MLflow.")
    mlflow_log(status, MLFLOW_LOG_PARAMS, params=params)


def _train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    probabilistic: bool = False,
    target_scaler: Optional[Any] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Runs one epoch of training and evaluation.

    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): Dataloader for training set.
        val_loader (DataLoader): Dataloader for validation set.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Torch device.
        epoch (int): Current epoch index.
        total_epochs (int): Total number of epochs.
        probabilistic (bool, optional): Use probabilistic model output. Defaults to False.
        target_scaler (Optional[Any], optional): Scaler to invert target transformation. Defaults to None.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Train and validation metrics (loss, rmse, r2).
    """
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}]", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if probabilistic:
            pred_dist = model(x, return_dist=True)
            loss = negative_log_likelihood(pred_dist, y)
            preds = pred_dist.mean
        else:
            preds = model(x)
            loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)

    y_true_np = np.expm1(y_true_np)
    y_pred_np = np.expm1(y_pred_np)

    rmse = float(np.sqrt(((y_pred_np - y_true_np) ** 2).mean()))
    r2 = float(r2_score(y_true_np, y_pred_np, multioutput="uniform_average"))
    logger.info(f"[Train] Epoch {epoch+1} — Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    train_metrics = {"loss": avg_loss, "rmse": rmse, "r2": r2}

    # ----- Validation -----
    model.eval()
    val_loss = 0.0
    y_val_true, y_val_pred = [], []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            if probabilistic:
                pred_dist = model(x_val, return_dist=True)
                loss = negative_log_likelihood(pred_dist, y_val)
                preds = pred_dist.mean
            else:
                preds = model(x_val)
                loss = criterion(preds, y_val)

            val_loss += loss.item()
            y_val_true.append(y_val.cpu().numpy())
            y_val_pred.append(preds.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    y_val_true_np = np.concatenate(y_val_true)
    y_val_pred_np = np.concatenate(y_val_pred)

    y_val_true_np = np.expm1(y_val_true_np)
    y_val_pred_np = np.expm1(y_val_pred_np)

    val_rmse = float(np.sqrt(((y_val_pred_np - y_val_true_np) ** 2).mean()))
    val_r2 = float(r2_score(y_val_true_np, y_val_pred_np, multioutput="uniform_average"))
    logger.info(f"[Val]   Epoch {epoch+1} — Loss: {avg_val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")

    val_metrics = {"loss": avg_val_loss, "rmse": val_rmse, "r2": val_r2}

    return train_metrics, val_metrics

def _update_histories(
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    """
    Appends current epoch metrics to their respective histories.

    Args:
        train_hist (Dict[str, List[float]]): Accumulated training metrics over epochs.
        val_hist (Dict[str, List[float]]): Accumulated validation metrics over epochs.
        train_metrics (Dict[str, float]): Current training epoch metrics.
        val_metrics (Dict[str, float]): Current validation epoch metrics.
    """
    for k in train_metrics:
        train_hist[k].append(train_metrics[k])
        val_hist[k].append(val_metrics[k])

def _save_best_model_and_log_metrics(
    model: nn.Module,
    optimizer: optim.Optimizer,
    checkpoint_path: str,
    mlflow_step: int,
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    mlflow_status: bool,
    log_model: bool
) -> None:
    """
    Saves the best model and logs the current metrics to MLflow.

    Args:
        model (nn.Module): The model being trained.
        optimizer (optim.Optimizer): Optimizer tied to the model.
        checkpoint_path (str): Path to store the checkpoint file.
        mlflow_step (int): Current training step or epoch used in MLflow logging.
        train_hist (Dict[str, List[float]]): Running list of training metrics.
        val_hist (Dict[str, List[float]]): Running list of validation metrics.
        train_metrics (Dict[str, float]): Metrics from this epoch's training.
        val_metrics (Dict[str, float]): Metrics from this epoch's validation.
        mlflow_status (bool): Whether MLflow is actively logging.
        log_model (bool): If True, save and log the model as the best checkpoint.
    """
    if log_model:
        logger.info("New best model found, saving...")

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": mlflow_step,
            "optimiser_state_dict": optimizer.state_dict(),
            **{f"train_{k}_history": v for k, v in train_hist.items()},
            **{f"val_{k}_history": v for k, v in val_hist.items()},
        }, checkpoint_path)

        mlflow_log(mlflow_status, MLFLOW_LOG_ARTIFACT, local_path=checkpoint_path)

    for k in train_metrics:
        mlflow_log(mlflow_status, MLFLOW_LOG_METRIC, key=f"train_{k}", value=train_metrics[k], step=mlflow_step)
        mlflow_log(mlflow_status, MLFLOW_LOG_METRIC, key=f"validation_{k}", value=val_metrics[k], step=mlflow_step)

def _finalize_training(
    train_hist: Dict[str, List[float]],
    val_hist: Dict[str, List[float]],
    mlflow_status: bool,
    artifact_dir: str,
    model: nn.Module,
    model_name: str,
    feature_scaler: Optional[StandardScaler],
    target_scaler: Optional[StandardScaler]
) -> None:
    """
    Wraps up training by logging final results and saving artifacts.

    Args:
        train_hist (Dict[str, List[float]]): Full training metric history.
        val_hist (Dict[str, List[float]]): Full validation metric history.
        mlflow_status (bool): Whether to log to MLflow.
        artifact_dir (str): Directory where artifacts like plots are stored.
        model (nn.Module): The trained model.
        model_name (str): Name to register the model in MLflow.
        feature_scaler (Optional[StandardScaler]): Feature scaler object.
        target_scaler (Optional[StandardScaler]): Target scaler object.
    """
    plot_training_curves(
        train_loss=train_hist["loss"],
        val_loss=val_hist["loss"],
        train_rmse=train_hist["rmse"],
        val_rmse=val_hist["rmse"],
        train_r2=train_hist["r2"],
        val_r2=val_hist["r2"]
    )
    logger.info("Plotted training curves.")

    mlflow_log(mlflow_status, MLFLOW_LOG_DICT, dictionary={}, artifact_file="configs/params.json")
    mlflow_log(mlflow_status, MLFLOW_LOG_ARTIFACTS, local_dir=artifact_dir, artifact_path="logs")

    mlflow_pytorch_call(
        mlflow_status,
        pytorch_function=MLFLOW_LOG_MODEL,
        pytorch_model=model,
        artifact_path="model",
        registered_model_name=model_name
    )

    if feature_scaler is not None:
        joblib.dump(feature_scaler, "artifacts/feature_scaler.pkl")
        mlflow.log_artifact("artifacts/feature_scaler.pkl")

    if target_scaler is not None:
        joblib.dump(target_scaler, "artifacts/target_scaler.pkl")
        mlflow.log_artifact("artifacts/target_scaler.pkl")

    if mlflow_status:
        artifact_uri = mlflow.get_artifact_uri()
        mlflow_log(mlflow_status, "log_params", params={"artifact_uri": artifact_uri})
        mlflow.end_run()
        logger.info("MLflow run ended.")


def walk_forward_validate(
    df: pd.DataFrame,
    cfg: DictConfig,
    device: torch.device,
    target_scaler: Optional[Any] = None,
) -> None:
    """
    Implements walk-forward validation over multiple rolling folds.

    Args:
        df (pd.DataFrame): Full historical time-series data.
        cfg (DictConfig): Config with fold settings, model, and training parameters.
        device (torch.device): Device to run training on.
        target_scaler (Optional[Any]): Optional scaler to reverse transforms.

    Returns:
        None
    """
    seq_len = cfg.data.seq_len
    pred_len = cfg.data.pred_len
    val_window = cfg.walk_forward.val_window
    n_folds = cfg.walk_forward.folds
    epochs = cfg.walk_forward.epochs_per_fold
    target = cfg.data.target
    feature_cols = df.columns.drop(target)

    total_hours = len(df)
    fold_stride = val_window

    for fold in range(n_folds):
        print(f"\n[Fold {fold+1}/{n_folds}]")

        train_end = total_hours - (n_folds - fold) * fold_stride
        val_start = train_end
        val_end = val_start + val_window

        if val_end + pred_len > total_hours:
            print("Not enough data left for prediction window. Skipping fold.")
            continue

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end + pred_len].copy()

        if cfg.data.use_scaling:
            feature_scaler = StandardScaler().fit(train_df[feature_cols])
            target_scaler  = StandardScaler().fit(train_df[[target]])

            train_df[feature_cols] = feature_scaler.transform(train_df[feature_cols])
            train_df[target]       = target_scaler.transform(train_df[[target]])
            val_df[feature_cols]   = feature_scaler.transform(val_df[feature_cols])
            val_df[target]         = target_scaler.transform(val_df[[target]])
        else:
            feature_scaler = target_scaler = None

        train_dataset = TimeSeriesDataset(train_df, target, seq_len, pred_len)
        val_dataset = TimeSeriesDataset(val_df, target, seq_len, pred_len)

        train_loader = DataLoader(train_dataset, batch_size=cfg.model_cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.model_cfg.batch_size, shuffle=False)

        model = create_model(
            name=cfg.model_cfg.model_type,
            input_size=len(feature_cols),
            output_size=pred_len,
            hidden_size=cfg.model_cfg.hidden_size,
            num_layers=cfg.model_cfg.num_layers,
            num_channels=cfg.model_cfg.num_channels
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model_cfg.lr)
        criterion = torch.nn.MSELoss()

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            cfg=deepcopy(cfg),
            scheduler=None
        )

        loss, rmse, r2 = evaluate(model, val_loader, criterion, device, target_scaler=target_scaler)
        print(f"[Fold {fold+1}] RMSE: {rmse:.4f}, R²: {r2:.4f}")


