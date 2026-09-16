"""Historical dam-failure Transformer training and prediction utilities."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

FEATURE_COLUMNS = ["dam_height_m", "water_volume_mcm", "water_height_m", "breach_width_m"]
TARGET_COLUMN = "peak_outflow_m3s"


def load_historical_records(source: str | bytes | io.BytesIO) -> pd.DataFrame:
    """Extract complete, observed numeric fields from the supplied DATABASE sheet."""
    raw = pd.read_excel(source, sheet_name="DATABASE", header=None)
    values = raw.iloc[10:].copy()
    selected = values.iloc[:, [22, 27, 28, 33, 36]].copy()
    selected.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    selected = selected.apply(pd.to_numeric, errors="coerce").dropna()
    return selected[(selected > 0).all(axis=1)].reset_index(drop=True)


class HistoricalDamTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, 16)
        layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.05, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.embedding(x.unsqueeze(-1))))


def train_historical_model(records: pd.DataFrame, seed: int = 7):
    """Fit the Transformer solely to complete historical records—no synthetic data."""
    if len(records) < 12:
        raise ValueError("At least 12 complete historical records are required.")
    x = records[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = np.log1p(records[TARGET_COLUMN].to_numpy(dtype=float)).reshape(-1, 1)
    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaled, y_scaled = x_scaler.fit_transform(x), y_scaler.fit_transform(y)
    torch.manual_seed(seed)
    model = HistoricalDamTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=0.002)
    inputs, labels = torch.tensor(x_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32)
    model.train()
    for _ in range(350):
        optimizer.zero_grad()
        nn.MSELoss()(model(inputs), labels).backward()
        optimizer.step()
    model.eval()
    return model, x_scaler, y_scaler


def predict_peak_outflow(model, x_scaler, y_scaler, inputs: dict[str, float]) -> float:
    x = np.array([[inputs[column] for column in FEATURE_COLUMNS]], dtype=float)
    with torch.no_grad():
        prediction = model(torch.tensor(x_scaler.transform(x), dtype=torch.float32)).numpy()
    return float(np.expm1(y_scaler.inverse_transform(prediction)[0, 0]))
