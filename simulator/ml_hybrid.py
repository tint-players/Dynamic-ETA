from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .ml_baselines import ETAMetrics, evaluate_eta_predictor
from .ml_features import CONTEXT_FEATURE_NAMES, GRAPH_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES
from .ml_gnn import DEFAULT_GNN_HIDDEN_SIZE, DEFAULT_GNN_LAYERS, GraphSAGEEncoder, _edge_index_tensor
from .ml_lstm import DEFAULT_HUBER_DELTA_S, DEFAULT_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_LAYERS, huber_eta_loss
from .ml_samples import MLModelInputs, MLTrainingSample


@dataclass(frozen=True, slots=True)
class TorchHybridBatch:
    x_seq: Tensor
    x_graph: Tensor
    edge_index: Tensor
    current_node_index: Tensor
    route_mask: Tensor
    x_context: Tensor
    y: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.y.shape[0])


def collate_hybrid_training_samples(
    samples: Sequence[MLTrainingSample],
    *,
    device: torch.device | str | None = None,
) -> TorchHybridBatch:
    if not samples:
        raise ValueError("Cannot collate an empty hybrid training batch")

    first = samples[0].inputs
    node_ids = first.node_ids
    edge_index = first.edge_index
    node_count = len(node_ids)
    for sample in samples:
        inputs = sample.inputs
        if inputs.node_ids != node_ids:
            raise ValueError("All hybrid samples in a batch must use the same node ordering")
        if inputs.edge_index != edge_index:
            raise ValueError("All hybrid samples in a batch must use the same message-passing topology")
        if len(inputs.x_graph) != node_count or len(inputs.route_mask) != node_count:
            raise ValueError("Hybrid sample node dimensions do not match node_ids")
        if not 0 <= inputs.current_node_index < node_count:
            raise ValueError("Hybrid sample current_node_index is outside node_ids")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return TorchHybridBatch(
        x_seq=torch.tensor(
            [sample.inputs.x_seq for sample in samples],
            dtype=torch.float32,
            device=target_device,
        ),
        x_graph=torch.tensor(
            [sample.inputs.x_graph for sample in samples],
            dtype=torch.float32,
            device=target_device,
        ),
        edge_index=_edge_index_tensor(edge_index, target_device),
        current_node_index=torch.tensor(
            [sample.inputs.current_node_index for sample in samples],
            dtype=torch.long,
            device=target_device,
        ),
        route_mask=torch.tensor(
            [sample.inputs.route_mask for sample in samples],
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


class HybridLSTMGraphSAGEETARegressor(nn.Module):
    """Hybrid ETA model combining temporal train history and railway graph state.

    The temporal branch is the agreed two-layer LSTM over the inference-safe
    60-second sequence. The spatial branch reuses the tested GraphSAGE encoder
    over canonical track-block nodes and bidirectional message-passing edges.
    Graph output is pooled into current-node and configured-route context,
    compressed to one graph representation, then fused with the LSTM state and
    current train context. The output is constrained to non-negative seconds.
    """

    def __init__(
        self,
        *,
        sequence_feature_count: int = len(SEQUENCE_FEATURE_NAMES),
        graph_feature_count: int = len(GRAPH_FEATURE_NAMES),
        context_feature_count: int = len(CONTEXT_FEATURE_NAMES),
        lstm_hidden_size: int = DEFAULT_LSTM_HIDDEN_SIZE,
        lstm_layers: int = DEFAULT_LSTM_LAYERS,
        gnn_hidden_size: int = DEFAULT_GNN_HIDDEN_SIZE,
        gnn_layers: int = DEFAULT_GNN_LAYERS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if sequence_feature_count <= 0 or graph_feature_count <= 0 or context_feature_count <= 0:
            raise ValueError("Hybrid feature dimensions must be positive")
        if lstm_hidden_size <= 0 or lstm_layers <= 0 or gnn_hidden_size <= 0 or gnn_layers <= 0:
            raise ValueError("Hybrid hidden sizes and layer counts must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.sequence_feature_count = sequence_feature_count
        self.graph_feature_count = graph_feature_count
        self.context_feature_count = context_feature_count
        self.lstm_hidden_size = lstm_hidden_size
        self.gnn_hidden_size = gnn_hidden_size

        self.lstm = nn.LSTM(
            input_size=sequence_feature_count,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.graph_encoder = GraphSAGEEncoder(
            graph_feature_count=graph_feature_count,
            hidden_size=gnn_hidden_size,
            num_layers=gnn_layers,
            dropout=dropout,
        )
        self.graph_pool = nn.Sequential(
            nn.Linear(gnn_hidden_size * 2, gnn_hidden_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size + gnn_hidden_size + context_feature_count, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x_seq: Tensor,
        x_graph: Tensor,
        edge_index: Tensor,
        current_node_index: Tensor,
        route_mask: Tensor,
        x_context: Tensor,
    ) -> Tensor:
        if x_seq.ndim != 3:
            raise ValueError("x_seq must have shape [batch, time, features]")
        if x_graph.ndim != 3:
            raise ValueError("x_graph must have shape [batch, nodes, features]")
        if current_node_index.ndim != 1:
            raise ValueError("current_node_index must have shape [batch]")
        if route_mask.ndim != 2:
            raise ValueError("route_mask must have shape [batch, nodes]")
        if x_context.ndim != 2:
            raise ValueError("x_context must have shape [batch, context_features]")

        batch_size = x_seq.shape[0]
        node_count = x_graph.shape[1]
        if x_graph.shape[0] != batch_size or current_node_index.shape[0] != batch_size:
            raise ValueError("Hybrid branch batch dimensions must match")
        if route_mask.shape != (batch_size, node_count):
            raise ValueError("Hybrid route_mask shape does not match graph nodes")
        if x_context.shape != (batch_size, self.context_feature_count):
            raise ValueError("Unexpected hybrid context shape")
        if x_seq.shape[-1] != self.sequence_feature_count:
            raise ValueError("Unexpected hybrid sequence feature count")
        if x_graph.shape[-1] != self.graph_feature_count:
            raise ValueError("Unexpected hybrid graph feature count")
        if torch.any(current_node_index < 0) or torch.any(current_node_index >= node_count):
            raise ValueError("current_node_index contains an invalid node")

        _, (hidden, _) = self.lstm(x_seq)
        temporal_embedding = hidden[-1]

        node_embeddings = self.graph_encoder(x_graph, edge_index)
        batch_indices = torch.arange(batch_size, device=x_graph.device)
        current_embedding = node_embeddings[batch_indices, current_node_index]

        mask = route_mask.to(dtype=node_embeddings.dtype).unsqueeze(-1)
        route_denominator = mask.sum(dim=1).clamp_min(1.0)
        route_embedding = (node_embeddings * mask).sum(dim=1) / route_denominator
        graph_embedding = self.graph_pool(torch.cat((current_embedding, route_embedding), dim=-1))

        fused = torch.cat((temporal_embedding, graph_embedding, x_context), dim=-1)
        return self.head(fused).squeeze(-1)


def train_hybrid_epoch(
    model: HybridLSTMGraphSAGEETARegressor,
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
        raise ValueError("Cannot train on an empty hybrid sample set")
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
        batch = collate_hybrid_training_samples(batch_samples, device=target_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch.x_seq,
            batch.x_graph,
            batch.edge_index,
            batch.current_node_index,
            batch.route_mask,
            batch.x_context,
        )
        loss = huber_eta_loss(prediction, batch.y, delta_s=huber_delta_s)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        total_weighted_loss += float(loss.detach().cpu()) * batch.batch_size
        total_samples += batch.batch_size

    return total_weighted_loss / total_samples


class TorchHybridETAPredictor:
    """Live-safe adapter exposing the hybrid model via the common ETA contract."""

    def __init__(
        self,
        model: HybridLSTMGraphSAGEETARegressor,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)

    def predict_seconds(self, inputs: MLModelInputs) -> float:
        self.model.eval()
        with torch.no_grad():
            x_seq = torch.tensor([inputs.x_seq], dtype=torch.float32, device=self.device)
            x_graph = torch.tensor([inputs.x_graph], dtype=torch.float32, device=self.device)
            edge_index = _edge_index_tensor(inputs.edge_index, self.device)
            current_node_index = torch.tensor([inputs.current_node_index], dtype=torch.long, device=self.device)
            route_mask = torch.tensor([inputs.route_mask], dtype=torch.float32, device=self.device)
            x_context = torch.tensor([inputs.x_context], dtype=torch.float32, device=self.device)
            prediction = self.model(
                x_seq,
                x_graph,
                edge_index,
                current_node_index,
                route_mask,
                x_context,
            )[0]
        return float(prediction.cpu())


def evaluate_hybrid_model(
    model: HybridLSTMGraphSAGEETARegressor,
    samples: Iterable[MLTrainingSample],
    *,
    device: torch.device | str | None = None,
) -> ETAMetrics:
    return evaluate_eta_predictor(TorchHybridETAPredictor(model, device=device), samples)
