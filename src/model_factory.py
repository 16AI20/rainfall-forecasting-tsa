"""
model_factory.py

Factory function to instantiate supported time series forecasting models
based on a string identifier. This abstraction allows clean and flexible
model selection during configuration or runtime.
"""

from typing import Any
from torch.nn import Module

from .model import LSTMForecaster, TCNForecaster, DeepARForecaster


def create_model(
    name: str,
    input_size: int,
    output_size: int,
    **kwargs: Any
) -> Module:
    """
    Factory method to create time series forecasters.

    Supported model types:
    - "lstm": LSTM-based forecaster
    - "tcn" : Temporal Convolutional Network forecaster
    - "deepar": Probabilistic DeepAR forecaster

    Args:
        name: Model identifier ("lstm", "tcn", "deepar").
        input_size: Number of input features (e.g., 5 for multivariate).
        output_size: Number of steps to forecast (prediction length).
        **kwargs: Additional model-specific arguments, such as:
            - hidden_size (int): For LSTM/DeepAR.
            - num_layers (int): For LSTM/DeepAR.
            - num_channels (List[int]): For TCN.
            - kernel_size (int): For TCN.

    Returns:
        Instantiated PyTorch model (`nn.Module`).

    Raises:
        ValueError: If an unsupported model name is passed.
    """
    name = name.lower()

    if name == "lstm":
        return LSTMForecaster(
            input_size=input_size,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            output_size=output_size,
        )

    elif name == "tcn":
        return TCNForecaster(
            input_size=input_size,
            output_size=output_size,
            num_channels=kwargs.get("num_channels", [32, 32, 32]),
            kernel_size=kwargs.get("kernel_size", 2),
        )

    elif name == "deepar":
        return DeepARForecaster(
            input_size=input_size,
            hidden_size=kwargs.get("hidden_size", 64),
            num_layers=kwargs.get("num_layers", 2),
            output_size=output_size,
        )

    else:
        raise ValueError(f"Unknown model name: {name!r}")
