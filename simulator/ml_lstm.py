from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ml_baselines import ETAMetrics, evaluate_eta_predictor
from .ml_features import CONTEXT_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES
from .ml_samples import MLModelInputs, MLTrainingSample


DEFAULT_LSTM_HIDDEN_SIZE = 128
DEFAULT_LSTM_LAYERS = 2
DEFAULT_HUBER_DELTA_S = 60.0


@dataclass(frozen=True, slots=True)
class TorchLSTMBatch:
    x_seq: Tensor
    x_context: Tensor
    y: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.y.shape[0])


def collate_lstm_training_samples(
    samples: Sequence[MLTrainingSample],
    *,
    device: torch.device | str | None = None,
) -> TorchLSTMBatch:
    if not samples:
        raise ValueError("Cannot collate an empty LSTM training batch")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return TorchLSTMBatch(
        x_seq=torch.tensor(
            [sample.inputs.x_seq for sample in samples],
            dtype=torch.float32,
            device=target_device,
        ),
        x_context=torch.tensor(
            [sample.inputs.x_context for sample in samples],
            dtype=torch.float32,
            device=target_device,
        ),
        y=torch.tensor([sample.y for sample in samples], dtype=torch.float32, device=target_device),
    )


class LSTMETARegressor(nn.Module):
    """LSTM-only ETA benchmark before adding the railway graph encoder.

    The model consumes the exact inference-safe 60-second sequence plus current
    train context. It deliberately does not read graph features, so it provides
    a clean benchmark for measuring the later contribution of GraphSAGE.
    """

    def __init__(
        self,
        *,
        sequence_feature_count: int = len(SEQUENCE_FEATURE_NAMES),
        context_feature_count: int = len(CONTEXT_FEATURE_NAMES),
        hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE,
        num_layers: int = DEFAULT_LSTM_LAYERS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.sequence_feature_count = sequence_feature_count
        self.context_feature_count = context_feature_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=sequence_feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + context_feature_count, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x_seq: Tensor, x_context: Tensor) -> Tensor:
        if x_seq.ndim != 3:
            raise ValueError("x_seq must have shape [batch, time, features]")
        if x_context.ndim != 2:
            raise ValueError("x_context must have shape [batch, context_features]")
        if x_seq.shape[0] != x_context.shape[0]:
            raise ValueError("x_seq and x_context batch dimensions must match")
        if x_seq.shape[-1] != self.sequence_feature_count:
            raise ValueError("Unexpected LSTM sequence feature count")
        if x_context.shape[-1] != self.context_feature_count:
            raise ValueError("Unexpected LSTM context feature count")

        _, (hidden, _) = self.lstm(x_seq)
        final_hidden = hidden[-1]
        fused = torch.cat((final_hidden, x_context), dim=-1)
        return self.head(fused).squeeze(-1)


def huber_eta_loss(prediction_s: Tensor, target_s: Tensor, *, delta_s: float = DEFAULT_HUBER_DELTA_S) -> Tensor:
    if delta_s <= 0:
        raise ValueError("delta_s must be positive")
    return F.huber_loss(prediction_s, target_s, delta=delta_s)


def train_lstm_epoch(
    model: LSTMETARegressor,
    samples: Iterable[MLTrainingSample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 128,
    device: torch.device | str | None = None,
    shuffle_seed: int = 42,
    huber_delta_s: float = DEFAULT_HUBER_DELTA_S,
    gradient_clip_norm: float = 1.0,
) -> float:
    materialized = list(samples)
    if not materialized:
        raise ValueError("Cannot train on an empty sample set")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")

    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(target_device)
    model.train()

    indices = list(range(len(materialized)))
    Random(shuffle_seed).shuffle(indices)
    total_weighted_loss = 0.0
    total_samples = 0

    for start in range(0, len(indices), batch_size):
        batch_samples = [materialized[index] for index in indices[start:start + batch_size]]
        batch = collate_lstm_training_samples(batch_samples, device=target_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch.x_seq, batch.x_context)
        loss = huber_eta_loss(prediction, batch.y, delta_s=huber_delta_s)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        total_weighted_loss += float(loss.detach().cpu()) * batch.batch_size
        total_samples += batch.batch_size

    return total_weighted_loss / total_samples


class TorchLSTMETAPredictor:
    """Adapter exposing a trained LSTM through the common ETA predictor contract."""

    def __init__(self, model: LSTMETARegressor, *, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)

    def predict_seconds(self, inputs: MLModelInputs) -> float:
        self.model.eval()
        with torch.no_grad():
            x_seq = torch.tensor([inputs.x_seq], dtype=torch.float32, device=self.device)
            x_context = torch.tensor([inputs.x_context], dtype=torch.float32, device=self.device)
            prediction = self.model(x_seq, x_context)[0]
        return float(prediction.cpu())


def evaluate_lstm_model(
    model: LSTMETARegressor,
    samples: Iterable[MLTrainingSample],
    *,
    device: torch.device | str | None = None,
) -> ETAMetrics:
    return evaluate_eta_predictor(TorchLSTMETAPredictor(model, device=device), samples)
