"""
model.py

This module defines deep learning architectures for time series forecasting:
- LSTMForecaster: Standard LSTM-based regression model
- TCNForecaster: Temporal Convolutional Network model
- DeepARForecaster: Probabilistic RNN model (GRU + Normal output distribution)
"""

from typing import List

import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn.utils import weight_norm


class LSTMForecaster(nn.Module):
    """
    LSTM-based time series forecasting model.

    Args:
        input_size: Number of input features.
        hidden_size: Size of hidden layers in LSTM.
        num_layers: Number of LSTM layers.
        output_size: Number of steps to forecast.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Forecast tensor of shape (batch_size, output_size)
        """
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_size)
        out = self.fc(h_n[-1])      # last layer's hidden state
        return out


class TCNBlock(nn.Module):
    """
    Single block in the Temporal Convolutional Network (TCN).

    Args:
        in_channels: Input channel size.
        out_channels: Output channel size.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)

        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        out = self.net(x)
        if self.conv.padding[0] > 0:
            out = out[:, :, :-self.conv.padding[0]]
        return out


class TCNForecaster(nn.Module):
    """
    Temporal Convolutional Network for sequence forecasting.

    Args:
        input_size: Number of input features.
        output_size: Number of prediction steps.
        num_channels: List of output channels per TCN block.
        kernel_size: Size of the 1D convolution kernels.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: List[int],
        kernel_size: int = 2
    ) -> None:
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=2 ** i))

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, output_size)
        """
        x = x.transpose(1, 2)  # -> (batch, input_size, seq_len)
        y = self.network(x)
        out = self.linear(y[:, :, -1])  # use last time step
        return out


class DeepARForecaster(nn.Module):
    """
    DeepAR-style probabilistic GRU model with Gaussian output.

    Args:
        input_size: Number of input features.
        hidden_size: GRU hidden size.
        num_layers: Number of GRU layers.
        output_size: Prediction length (forecast horizon).
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.proj_mu = nn.Linear(hidden_size, output_size)
        self.proj_sigma = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, return_dist: bool = False) -> torch.Tensor | D.Distribution:
        """
        Forward pass through GRU -> Normal(mu, sigma)

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            return_dist: If True, return full Normal distribution.

        Returns:
            - If return_dist=True: torch.distributions.Normal
            - Else: predicted mean tensor (batch, output_size)
        """
        out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_size)
        h = out[:, -1, :]     # last hidden state

        mu = self.proj_mu(h)                         # (batch, output_size)
        sigma = self.softplus(self.proj_sigma(h)) + 1e-6  # avoid zero

        if return_dist:
            return D.Normal(loc=mu, scale=sigma)
        return mu
