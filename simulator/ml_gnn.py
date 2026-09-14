from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .ml_baselines import ETAMetrics, evaluate_eta_predictor
from .ml_features import CONTEXT_FEATURE_NAMES, GRAPH_FEATURE_NAMES
from .ml_lstm import DEFAULT_HUBER_DELTA_S, huber_eta_loss
from .ml_samples import MLModelInputs, MLTrainingSample


DEFAULT_GNN_HIDDEN_SIZE = 128
DEFAULT_GNN_LAYERS = 3


@dataclass(frozen=True, slots=True)
class TorchGNNBatch:
    x_graph: Tensor
    edge_index: Tensor
    current_node_index: Tensor
    route_mask: Tensor
    x_context: Tensor
    y: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.y.shape[0])


def _edge_index_tensor(edge_index: tuple[tuple[int, ...], tuple[int, ...]], device: torch.device) -> Tensor:
    if len(edge_index) != 2:
        raise ValueError("edge_index must contain source and destination rows")
    if len(edge_index[0]) != len(edge_index[1]):
        raise ValueError("edge_index source and destination rows must have equal length")
    return torch.tensor(edge_index, dtype=torch.long, device=device)


def collate_gnn_training_samples(
    samples: Sequence[MLTrainingSample],
    *,
    device: torch.device | str | None = None,
) -> TorchGNNBatch:
    if not samples:
        raise ValueError("Cannot collate an empty GNN training batch")

    first = samples[0].inputs
    node_ids = first.node_ids
    edge_index = first.edge_index
    node_count = len(node_ids)
    for sample in samples:
        inputs = sample.inputs
        if inputs.node_ids != node_ids:
            raise ValueError("All GNN samples in a batch must use the same node ordering")
        if inputs.edge_index != edge_index:
            raise ValueError("All GNN samples in a batch must use the same message-passing topology")
        if len(inputs.x_graph) != node_count or len(inputs.route_mask) != node_count:
            raise ValueError("GNN sample node dimensions do not match node_ids")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return TorchGNNBatch(
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


class MeanGraphSAGELayer(nn.Module):
    """Dependency-light mean-aggregation GraphSAGE layer.

    The layer consumes the canonical shared ``edge_index`` already produced by
    ``MLGraphTopologyBuilder``. That topology is bidirectional for message
    passing while the separate operational topology remains available for
    railway semantics and auditing.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        if input_size <= 0 or output_size <= 0:
            raise ValueError("GraphSAGE dimensions must be positive")
        self.self_linear = nn.Linear(input_size, output_size)
        self.neighbour_linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, nodes, features]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")

        batch_size, node_count, feature_count = x.shape
        if feature_count != self.self_linear.in_features:
            raise ValueError("Unexpected GraphSAGE input feature count")

        if edge_index.numel() == 0:
            neighbour_mean = torch.zeros_like(x)
        else:
            source = edge_index[0]
            destination = edge_index[1]
            if torch.any(source < 0) or torch.any(destination < 0):
                raise ValueError("edge_index cannot contain negative node indices")
            if torch.any(source >= node_count) or torch.any(destination >= node_count):
                raise ValueError("edge_index contains a node index outside x")

            neighbour_sum = torch.zeros_like(x)
            neighbour_sum.index_add_(1, destination, x.index_select(1, source))
            degree = torch.bincount(destination, minlength=node_count).to(dtype=x.dtype, device=x.device)
            degree = degree.clamp_min(1.0).view(1, node_count, 1)
            neighbour_mean = neighbour_sum / degree

        return self.self_linear(x) + self.neighbour_linear(neighbour_mean)


class GraphSAGEEncoder(nn.Module):
    """Three-layer railway graph encoder used before hybrid LSTM fusion."""

    def __init__(
        self,
        *,
        graph_feature_count: int = len(GRAPH_FEATURE_NAMES),
        hidden_size: int = DEFAULT_GNN_HIDDEN_SIZE,
        num_layers: int = DEFAULT_GNN_LAYERS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if graph_feature_count <= 0:
            raise ValueError("graph_feature_count must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.graph_feature_count = graph_feature_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        layers: list[MeanGraphSAGELayer] = []
        for index in range(num_layers):
            input_size = graph_feature_count if index == 0 else hidden_size
            layers.append(MeanGraphSAGELayer(input_size, hidden_size))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_graph: Tensor, edge_index: Tensor) -> Tensor:
        if x_graph.ndim != 3:
            raise ValueError("x_graph must have shape [batch, nodes, features]")
        if x_graph.shape[-1] != self.graph_feature_count:
            raise ValueError("Unexpected graph feature count")

        hidden = x_graph
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index)
            hidden = self.activation(hidden)
            if index < len(self.layers) - 1:
                hidden = self.dropout(hidden)
        return hidden


class GraphSAGEETARegressor(nn.Module):
    """Graph-only ETA benchmark using current-node and route-context embeddings."""

    def __init__(
        self,
        *,
        graph_feature_count: int = len(GRAPH_FEATURE_NAMES),
        context_feature_count: int = len(CONTEXT_FEATURE_NAMES),
        hidden_size: int = DEFAULT_GNN_HIDDEN_SIZE,
        num_layers: int = DEFAULT_GNN_LAYERS,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.context_feature_count = context_feature_count
        self.encoder = GraphSAGEEncoder(
            graph_feature_count=graph_feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2 + context_feature_count, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        x_graph: Tensor,
        edge_index: Tensor,
        current_node_index: Tensor,
        route_mask: Tensor,
        x_context: Tensor,
    ) -> Tensor:
        if current_node_index.ndim != 1:
            raise ValueError("current_node_index must have shape [batch]")
        if route_mask.ndim != 2:
            raise ValueError("route_mask must have shape [batch, nodes]")
        if x_context.ndim != 2:
            raise ValueError("x_context must have shape [batch, context_features]")
        batch_size, node_count, _ = x_graph.shape
        if current_node_index.shape[0] != batch_size or route_mask.shape != (batch_size, node_count):
            raise ValueError("GNN batch dimensions do not match")
        if x_context.shape != (batch_size, self.context_feature_count):
            raise ValueError("Unexpected GNN context shape")
        if torch.any(current_node_index < 0) or torch.any(current_node_index >= node_count):
            raise ValueError("current_node_index contains an invalid node")

        node_embeddings = self.encoder(x_graph, edge_index)
        batch_indices = torch.arange(batch_size, device=x_graph.device)
        current_embedding = node_embeddings[batch_indices, current_node_index]

        mask = route_mask.to(dtype=node_embeddings.dtype).unsqueeze(-1)
        route_denominator = mask.sum(dim=1).clamp_min(1.0)
        route_embedding = (node_embeddings * mask).sum(dim=1) / route_denominator

        fused = torch.cat((current_embedding, route_embedding, x_context), dim=-1)
        return self.head(fused).squeeze(-1)


def train_gnn_epoch(
    model: GraphSAGEETARegressor,
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
        batch = collate_gnn_training_samples(batch_samples, device=target_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
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


class TorchGraphSAGEETAPredictor:
    """Adapter exposing a trained GraphSAGE model through the ETA predictor contract."""

    def __init__(self, model: GraphSAGEETARegressor, *, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)

    def predict_seconds(self, inputs: MLModelInputs) -> float:
        self.model.eval()
        with torch.no_grad():
            x_graph = torch.tensor([inputs.x_graph], dtype=torch.float32, device=self.device)
            edge_index = _edge_index_tensor(inputs.edge_index, self.device)
            current_node_index = torch.tensor([inputs.current_node_index], dtype=torch.long, device=self.device)
            route_mask = torch.tensor([inputs.route_mask], dtype=torch.float32, device=self.device)
            x_context = torch.tensor([inputs.x_context], dtype=torch.float32, device=self.device)
            prediction = self.model(
                x_graph,
                edge_index,
                current_node_index,
                route_mask,
                x_context,
            )[0]
        return float(prediction.cpu())


def evaluate_gnn_model(
    model: GraphSAGEETARegressor,
    samples: Iterable[MLTrainingSample],
    *,
    device: torch.device | str | None = None,
) -> ETAMetrics:
    return evaluate_eta_predictor(TorchGraphSAGEETAPredictor(model, device=device), samples)
