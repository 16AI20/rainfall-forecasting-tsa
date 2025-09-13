"""
pipeline.py

Main forecasting pipeline using Hydra config. This script handles:
- Data loading and preprocessing
- Feature/target scaling
- Model creation (LSTM, TCN, or DeepAR)
- Training (standard or walk-forward)
- Evaluation (RMSE, R², MSE)

This entry point is Hydra-compatible and configurable via YAML.
"""

# Standard Library Imports
import logging
from typing import Optional

# Third-Party Libraries
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
import torch

from .data_loader import create_dataloaders
from .data_processor import train_test_split
from .model_factory import create_model
from .train import train_model, walk_forward_validate
from .evaluate import evaluate
from .general_utils import set_global_seed

from data_processor import (
    load_and_prepare_df,
    scale_features,
    log_transform_targets
)

logger = logging.getLogger(__name__)


class ForecastingPipeline:
    """
    Core class that encapsulates the forecasting pipeline:
    loading, preprocessing, model setup, training, and evaluation.

    Args:
        cfg: Hydra configuration object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        load_dotenv()
        self.cfg = cfg

        model_type = cfg.model_cfg.model_type.lower()
        cfg.model_cfg.probabilistic = model_type == "deepar"
        set_global_seed(cfg.seed)

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.target_scaler: Optional[StandardScaler] = None
        self.feature_scaler: Optional[StandardScaler] = None

    def setup_data(self) -> None:
        logger.info("Loading and preprocessing dataset...")
        self.df = load_and_prepare_df(self.cfg)

        train_df, val_df = train_test_split(self.df, split_date=self.cfg.data.split_date)

        train_df, val_df, self.feature_scaler, self.target_scaler = scale_features(
            train_df, val_df, self.cfg.data.target, self.cfg.data.use_scaling
        )

        train_df, val_df = log_transform_targets(train_df, val_df, self.cfg.data.target)

        self.train_loader, self.val_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            target=self.cfg.data.target,
            seq_len=self.cfg.data.seq_len,
            pred_len=self.cfg.data.pred_len,
            batch_size=self.cfg.model_cfg.batch_size,
        )

        self.input_size = len(self.df.columns)
        self.output_size = self.cfg.data.pred_len

    def setup_model(self) -> None:
        """
        Initialize model, optimizer, criterion, and learning rate scheduler.
        """
        logger.info("Instantiating model: %s", self.cfg.model_cfg.model_type)
        self.model = create_model(
            name=self.cfg.model_cfg.model_type,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.cfg.model_cfg.hidden_size,
            num_layers=self.cfg.model_cfg.num_layers,
            num_channels=self.cfg.model_cfg.num_channels
        ).to(self.device)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.model_cfg.lr
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.cfg.scheduler.mode,
            patience=self.cfg.scheduler.patience,
            factor=self.cfg.scheduler.factor
        )

    def train(self) -> None:
        """
        Train model using either standard train/val split or walk-forward validation.
        """
        if self.cfg.walk_forward.get("enabled", False):
            logger.info("Running walk-forward validation...")
            walk_forward_validate(
                self.df,
                self.cfg,
                self.device,
                target_scaler=self.target_scaler
            )
        else:
            logger.info("Running standard training loop...")
            train_model(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                scheduler=self.scheduler,
                feature_scaler=self.feature_scaler,
                target_scaler=self.target_scaler,
                cfg=self.cfg
            )

    def evaluate(self) -> None:
        """
        Evaluate the trained model on the validation set.
        """
        val_loss, val_rmse, val_r2 = evaluate(
            model=self.model,
            val_loader=self.val_loader,
            criterion=self.criterion,
            device=self.device,
            target_scaler=self.target_scaler
        )
        logger.info("[FINAL EVAL] Loss: %.4f | RMSE: %.4f | R²: %.4f", val_loss, val_rmse, val_r2)


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Entry point for running the forecasting pipeline.
    """
    logger.info("Starting forecasting pipeline...")
    logger.info("Hydra Config:\n%s", OmegaConf.to_yaml(cfg))

    pipeline = ForecastingPipeline(cfg)

    logger.info("Setting up data...")
    pipeline.setup_data()

    logger.info("Initializing model...")
    pipeline.setup_model()

    logger.info("Training model...")
    pipeline.train()

    logger.info("Evaluating model...")
    pipeline.evaluate()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
