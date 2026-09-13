"""
models.py -- the four neural sequence architectures compared in this study.

All four take the same input, a (batch, LOOKBACK, n_features) tensor of scaled
log-discharge plus cyclical day-of-year features, and emit a (batch, HORIZON)
tensor: a direct multi-horizon forecast. Keeping the input, output, loss and
training loop identical across architectures is what makes the comparison in
run_experiment.py a comparison of architectures rather than of harnesses.

A note on the bidirectional model: the GRU runs both directions over the
*input window*, which contains only observed history up to the forecast
origin. Nothing downstream of the origin is visible to it, so there is no
leakage of the values it is being asked to predict.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])       # last hidden state summarises the window


class GRUForecaster(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class BiGRUForecaster(nn.Module):
    """Bidirectional GRU. The final representation concatenates the forward
    pass's last step with the backward pass's first step, so both ends of the
    observed window contribute."""

    def __init__(self, n_features: int, horizon: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.GRU(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        fwd = out[:, -1, :self.hidden]
        bwd = out[:, 0, self.hidden:]
        return self.head(torch.cat([fwd, bwd], dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Transformer encoder over the observed window. Deliberately small
    (d_model 64, 2 layers, 4 heads) -- with ~2k training windows a
    larger configuration overfits badly, which is itself a finding worth
    reporting rather than tuning away."""

    def __init__(self, n_features: int, horizon: int, d_model: int = 64, nhead: int = 4,
                 layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.input_proj(x))
        h = self.encoder(h)
        return self.head(h.mean(dim=1))    # mean-pool across the window


MODEL_REGISTRY = {
    "LSTM": LSTMForecaster,
    "GRU": GRUForecaster,
    "BiGRU": BiGRUForecaster,
    "Transformer": TransformerForecaster,
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
