import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data_loader import load_hourly_weather_data
from src.time_series_dataset import TimeSeriesDataset
from src.model import LSTMForecaster, TCNForecaster, DeepARForecaster

# --- Test 1: Data Loader ---
def test_load_hourly_weather_data_columns():
    df = load_hourly_weather_data("data/aggregated/all_features_hourly.csv")
    expected_columns = {'humidity', 'rainfall', 'temperature', 'wind_direction', 'wind_speed', 'wind_direction_sin', 'wind_direction_cos'}
    assert expected_columns.issubset(set(df.columns)), f"Missing expected columns: {expected_columns - set(df.columns)}"
    assert not df.isnull().values.any(), "DataFrame contains missing values after loading"

# --- Test 2: Model Forward Pass ---
@pytest.mark.parametrize("model_class", [LSTMForecaster, TCNForecaster, DeepARForecaster])
def test_model_forward_pass_shape(model_class):
    batch_size = 4
    seq_len = 168
    input_size = 5
    pred_len = 24
    x = torch.randn(batch_size, seq_len, input_size)

    if model_class == TCNForecaster:
        model = model_class(input_size=input_size, output_size=pred_len, num_channels=[32, 32, 32])
    else:
        model = model_class(input_size=input_size, hidden_size=32, num_layers=2, output_size=pred_len)

    output = model(x)
    assert output.shape == (batch_size, pred_len), f"Unexpected output shape: {output.shape}"

# --- Test 3: Dataset Sample Shape ---
def test_time_series_dataset_item_shape():
    df = load_hourly_weather_data("data/aggregated/all_features_hourly.csv")
    target = "rainfall"
    seq_len = 168
    pred_len = 24
    dataset = TimeSeriesDataset(df, target=target, seq_len=seq_len, pred_len=pred_len)
    x, y = dataset[0]
    assert x.shape == (seq_len, df.shape[1] - 1), f"Unexpected input shape: {x.shape}"
    assert y.shape == (pred_len,), f"Unexpected target shape: {y.shape}"
