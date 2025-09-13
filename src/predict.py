import os
import joblib
import torch
import pandas as pd
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
from datetime import datetime

import hydra
from omegaconf import DictConfig

from .model import LSTMForecaster, TCNForecaster, DeepARForecaster
from .data_loader import load_hourly_weather_data
from .data_processor import train_test_split
from .time_series_dataset import TimeSeriesDataset
from .api_client import WeatherAPIClient

logger = logging.getLogger(__name__)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_and_preprocess_data(data_path, target, split_date):
    df = load_hourly_weather_data(data_path)
    df.index = pd.to_datetime(df.index)
    feature_cols = df.columns

    train_df, test_df = train_test_split(df, split_date=split_date)
    train_df, test_df = train_df.copy(), test_df.copy()

    train_df[target] = train_df[target].astype("float64")
    test_df[target] = test_df[target].astype("float64")
    train_df[feature_cols] = train_df[feature_cols].astype("float64")
    test_df[feature_cols] = test_df[feature_cols].astype("float64")

    return train_df, test_df, feature_cols

def apply_scaling(train_df, test_df, feature_cols, scaler_path):
    feature_scaler = joblib.load(scaler_path)
    feature_scaler.fit(train_df[feature_cols])
    train_df.loc[:, feature_cols] = feature_scaler.transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = feature_scaler.transform(test_df[feature_cols])
    return train_df, test_df, feature_scaler

def create_test_dataset(test_df, target, seq_len, pred_len):
    return TimeSeriesDataset(test_df, target=target, seq_len=seq_len, pred_len=pred_len)

def load_model_by_name(cfg, device, input_size, output_size):
    model_name = cfg.model_cfg.model_type.lower()
    model_path = cfg.model_cfg.model_path
    if model_name == 'lstm':
        model = LSTMForecaster(input_size, cfg.model_cfg.hidden_size, cfg.model_cfg.num_layers, output_size)
    elif model_name == 'tcn':
        model = TCNForecaster(input_size, output_size, cfg.model_cfg.num_channels)
    elif model_name == 'deepar':
        model = DeepARForecaster(input_size, cfg.model_cfg.hidden_size, cfg.model_cfg.num_layers, output_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict(model, dataset, device):
    if isinstance(dataset, torch.Tensor):
        last_seq = dataset.to(device)
    else:
        last_seq, _ = dataset[-1]
        last_seq = last_seq.unsqueeze(0).to(device)
    with torch.no_grad():
        forecast = model(last_seq).cpu().numpy().flatten()
    return forecast

def build_input_sequence(date: datetime, cfg):
    api_client = WeatherAPIClient()
    hourly_dfs = []
    for feat in cfg.predict.api_features:
        df = api_client.fetch_feature_history(feat, date)
        df = df.resample("1h").mean().interpolate(method="time").bfill().ffill()
        hourly_dfs.append(df.rename(columns={"value": feat}))
    merged = pd.concat(hourly_dfs, axis=1).dropna()
    if "wind-direction" in merged.columns:
        logger.debug("Applying circular encoding to wind-direction")
        merged["wind-dir_sin"] = np.sin(np.deg2rad(merged["wind-direction"]))
        merged["wind-dir_cos"] = np.cos(np.deg2rad(merged["wind-direction"]))
        merged.drop("wind-direction", axis=1, inplace=True)
    return merged.iloc[-(cfg.data.seq_len + cfg.data.pred_len):]

def prepare_for_predict(date: datetime, cfg, device):
    seq_df = build_input_sequence(date, cfg)
    assert len(seq_df) == (cfg.data.seq_len + cfg.data.pred_len), f"Expected {cfg.data.seq_len + cfg.data.pred_len} steps, got {len(seq_df)}"
    arr = seq_df.values.astype("float32")
    return torch.from_numpy(arr).unsqueeze(0).to(device)

def plot_predictions(model, dataset, device, cfg):
    logger.info("Plotting prediction samples...")
    model.eval()
    os.makedirs(cfg.logging.artifact_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), cfg.predict.num_samples)
    model_name = model.__class__.__name__
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        input_seq = x.unsqueeze(0).to(device)
        with torch.no_grad():
            if isinstance(model, DeepARForecaster):
                dist = model(input_seq, return_dist=True)
                pred = dist.mean.cpu().numpy().flatten()
            else:
                pred = model(input_seq).cpu().numpy().flatten()
        pred = np.expm1(pred)
        actual = np.expm1(y.numpy().flatten())
        plt.figure(figsize=(10, 4))
        plt.plot(range(dataset.seq_len), dataset.df[dataset.target].iloc[idx:idx + dataset.seq_len], label="Historical", color="blue")
        plt.plot(range(dataset.seq_len, dataset.seq_len + dataset.pred_len), actual, label="Actual", color="green")
        plt.plot(range(dataset.seq_len, dataset.seq_len + dataset.pred_len), pred, label="Predicted", color="red", linestyle="--")
        plt.axvline(x=dataset.seq_len - 1, color="black", linestyle=":", label="Prediction Start")
        plt.legend()
        plt.title(f"{model_name} - Sample {i+1} - Prediction vs Actual")
        plt.tight_layout()
        filepath = os.path.join(cfg.logging.artifact_dir, f"{model_name}_sample_{i+1}.png")
        plt.savefig(filepath)
        plt.close()
        logger.debug("Saved plot to: %s", filepath)

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    device = get_device()
    dt_str = cfg.predict.get("date")
    dt = datetime.strptime(dt_str, "%Y-%m-%d") if dt_str else datetime.today()

    tensor = prepare_for_predict(dt, cfg, device)
    model = load_model_by_name(
        cfg=cfg,
        device=device,
        input_size=tensor.shape[2],
        output_size=cfg.data.pred_len
    )

    forecast = predict(model=model, dataset=tensor, device=device)
    print("Forecast:", forecast)

if __name__ == "__main__":
    main()

    # df = load_hourly_weather_data(DATA_PATH)
    # df.index = pd.to_datetime(df.index)

    # target = "rainfall"

    # # Split train/val
    # _, val_df = train_test_split(df, split_date=SPLIT_DATE)

    # # Ensure numeric dtype
    # val_df[VAL_FEATURES] = val_df[VAL_FEATURES].astype("float64")

    # # Determine which features to scale (exclude target)
    # features_to_scale = [f for f in VAL_FEATURES if f != TARGET]

    # # Load and apply feature scaler
    # feature_scaler = joblib.load(os.path.join(SCALER_DIR, "feature_scaler.pkl"))
    # feature_scaler.fit(val_df[features_to_scale])
    # val_df.loc[:, features_to_scale] = feature_scaler.transform(val_df[features_to_scale])

    # val_df[target] =  np.log1p(val_df[target])

    # # Circular encode wind_direction if present
    # if "wind_direction" in val_df.columns:
    #     val_df["wind_dir_sin"] = np.sin(np.deg2rad(val_df["wind_direction"]))
    #     val_df["wind_dir_cos"] = np.cos(np.deg2rad(val_df["wind_direction"]))
    #     val_df.drop("wind_direction", axis=1, inplace=True)

    # # Define input features (exclude target)
    # input_features = [f for f in val_df.columns if f != TARGET]

    # # Build validation dataset
    # val_dataset = TimeSeriesDataset(val_df, target=TARGET, seq_len=SEQ_LEN, pred_len=PRED_LEN)

    # # Plot predictions
    # plot_predictions(
    #     model=model,
    #     dataset=val_dataset,
    #     device=device,
    #     num_samples=3,
    #     save_dir="images/plots"
    # )
