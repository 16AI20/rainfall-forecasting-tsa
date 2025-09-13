"""
general_utils.py

Utility functions shared across modules. Includes:

- Logging configuration setup from YAML
- MLflow initialization, logging, and run resumption utilities
- Seed setting for reproducibility
- Safe wrappers around MLflow’s PyTorch functions
"""

import logging
import logging.config
import os
import random
import time
from typing import Any, Optional, Tuple, List

import numpy as np
import torch
import yaml
import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient


# Configure module-level logger
logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str = "./conf/logging.yaml",
    default_level: int = logging.INFO,
    log_dir: Optional[str] = None
) -> None:
    """
    Set up logging configuration from a YAML file.

    If config file is missing or malformed, falls back to basic config.

    Args:
        logging_config_path: Path to logging YAML configuration.
        default_level: Fallback logging level if config fails.
        log_dir: Optional directory to redirect file handlers to.
    """
    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            for handler in log_config.get("handlers", {}).values():
                if "filename" in handler:
                    handler["filename"] = os.path.join(
                        log_dir, os.path.basename(handler["filename"])
                    )

        logging.config.dictConfig(log_config)
    except (FileNotFoundError, PermissionError) as file_err:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.warning("Logging config not found or unreadable. Using basic config.")
        logger.exception(file_err)
    except (yaml.YAMLError, ValueError, TypeError) as parse_err:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.warning("Error parsing logging config. Using basic config.")
        logger.exception(parse_err)


def set_global_seed(seed: int = 42) -> int:
    """
    Sets global seeds for reproducibility across random, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.

    Returns:
        The seed used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Global seed set to %d", seed)
    return seed


def _set_mlflow_tags() -> None:
    """
    Set MLflow tags from common environment variables.
    """
    def set_tag(env_var: str, tag_name: str = "") -> None:
        if env_var in os.environ:
            key = tag_name or env_var.lower()
            mlflow.set_tag(key, os.environ.get(env_var))

    set_tag("MLFLOW_HP_TUNING_TAG", "hptuning_tag")
    set_tag("JOB_UUID")
    set_tag("JOB_NAME")


def _start_new_run(base_run_name: str) -> str:
    """
    Starts a new MLflow run with a timestamped name.

    Args:
        base_run_name: Prefix to use in the run name.

    Returns:
        Run name used.
    """
    run_name = f"{base_run_name}-{int(time.time())}"
    mlflow.start_run(run_name=run_name)
    logger.info("Starting new run: %s", run_name)
    return run_name


def _resume_previous_run(
    client: MlflowClient,
    exp_name: str,
    base_run_name: str
) -> Tuple[Optional[Run], int]:
    """
    Attempt to resume the most recent matching MLflow run.

    Args:
        client: Initialized MLflow client.
        exp_name: Experiment name.
        base_run_name: Name prefix used to identify the run.

    Returns:
        Tuple of (resumed Run or None, max logged step).
    """
    experiment = client.get_experiment_by_name(exp_name)
    if not experiment:
        return None, 0

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '{base_run_name}-%'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None, 0

    run = runs[0]
    mlflow.start_run(run_id=run.info.run_id)
    logger.info("Resuming previous run: %s", run.info.run_name)
    max_step = _get_max_logged_step(client, run.info.run_id)

    return run, max_step


def _get_metric_keys(client: MlflowClient, run_id: str) -> List[str]:
    """
    Retrieve available metric keys from a run.

    Args:
        client: MLflow client.
        run_id: Run ID to inspect.

    Returns:
        List of metric keys.
    """
    try:
        run_data = client.get_run(run_id).data
        keys = getattr(run_data.metrics, "keys", lambda: [])()
        if keys:
            return list(keys)

        return [m.key for m in getattr(run_data.metrics, "__iter__", lambda: [])()]
    except Exception as e:
        logger.warning("Failed to extract metric keys: %s", e)
        return []


def _get_max_step_for_key(client: MlflowClient, run_id: str, key: str) -> Optional[int]:
    """
    Find the highest logged step for a given metric.

    Args:
        client: MLflow client.
        run_id: Run ID to check.
        key: Metric key to inspect.

    Returns:
        Max step or None.
    """
    try:
        history = client.get_metric_history(run_id, key)
        if history:
            return max(m.step for m in history)
    except Exception as e:
        logger.warning("Error retrieving metric history for '%s': %s", key, e)
    return None


def _get_max_logged_step(client: MlflowClient, run_id: str) -> int:
    """
    Return the highest step across all metrics in a run.

    Args:
        client: MLflow client.
        run_id: Run ID.

    Returns:
        Max step (0 if none found).
    """
    metric_keys = _get_metric_keys(client, run_id)
    max_steps = filter(None, (_get_max_step_for_key(client, run_id, k) for k in metric_keys))
    return max(max_steps, default=0)


def mlflow_init(
    tracking_uri: str,
    exp_name: str,
    run_name: str,
    setup_mlflow: bool = False,
    autolog: bool = False,
    resume: bool = False,
) -> Tuple[bool, Optional[Run], int]:
    """
    Initialize or resume MLflow experiment run.

    Args:
        tracking_uri: MLflow server URI.
        exp_name: Experiment name.
        run_name: Name or prefix of the run.
        setup_mlflow: Whether to enable tracking setup.
        autolog: Whether to enable MLflow autologging.
        resume: Whether to resume previous matching run.

    Returns:
        Tuple of (init_success, active run, max logged step).
    """
    init_success = False
    mlflow_run = None
    step_offset = 0

    if not setup_mlflow:
        return init_success, mlflow_run, step_offset

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        mlflow.enable_system_metrics_logging()

        if autolog:
            mlflow.autolog()

        if "MLFLOW_HPTUNING_TAG" in os.environ:
            run_name += "-hp"

        base_run_name = run_name
        client = MlflowClient()

        if resume:
            mlflow_run, step_offset = _resume_previous_run(client, exp_name, base_run_name)
            if not mlflow_run:
                run_name = _start_new_run(base_run_name)
        else:
            run_name = _start_new_run(base_run_name)

        _set_mlflow_tags()

        mlflow_run = mlflow.active_run()
        init_success = True

        logger.info("MLflow initialisation succeeded.")
        logger.info("Run UUID: %s", mlflow_run.info.run_id)

    except Exception as e:
        logger.error("MLflow initialisation failed: %s", e)

    return init_success, mlflow_run, step_offset


def mlflow_log(
    mlflow_init_status: bool,
    log_function: str,
    **kwargs
) -> None:
    """
    Generic logger for MLflow metrics, parameters, or artifacts.

    Args:
        mlflow_init_status: True if MLflow is initialized.
        log_function: Name of MLflow logging function (e.g., 'log_metric').
        **kwargs: Parameters for the MLflow function.
    """
    if mlflow_init_status:
        try:
            method = getattr(mlflow, log_function)
            method(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in method.__code__.co_varnames
                }
            )
        except Exception as error:
            logger.error("MLflow log failed: %s", error)


def mlflow_pytorch_call(
    mlflow_init_status: bool,
    pytorch_function: str,
    **kwargs
) -> Optional[Any]:
    """
    Wrapper for safely calling mlflow.pytorch methods.

    Args:
        mlflow_init_status: True if MLflow is active.
        pytorch_function: Function name in mlflow.pytorch (e.g., 'log_model').
        **kwargs: Arguments to pass.

    Returns:
        Whatever the method returns, or None.
    """
    if not mlflow_init_status:
        return None

    try:
        method = getattr(mlflow.pytorch, pytorch_function)
    except AttributeError as err:
        logger.error("mlflow.pytorch has no function '%s': %s", pytorch_function, err)
        return None

    try:
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in method.__code__.co_varnames
        }
        return method(**valid_kwargs)
    except Exception as err:
        logger.error("mlflow.pytorch.%s failed: %s", pytorch_function, err)
        return None
    
def negative_log_likelihood(y_pred_dist, y_true) -> torch.Tensor:
    """
    Computes the negative log-likelihood for Gaussian outputs.

    Args:
        y_pred_dist: A torch.distributions.Normal instance returned by the model.
        y_true: Ground truth target tensor.

    Returns:
        torch.Tensor: Negative log-likelihood scalar.
    """
    return -y_pred_dist.log_prob(y_true).mean()
