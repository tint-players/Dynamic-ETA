from __future__ import annotations

import math

import torch

from simulator.ml_features import CONTEXT_FEATURE_NAMES, GRAPH_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES
from simulator.ml_hybrid import (
    HybridLSTMGraphSAGEETARegressor,
    TorchHybridETAPredictor,
    collate_hybrid_training_samples,
    train_hybrid_epoch,
)
from simulator.ml_samples import MLModelInputs, MLTrainingSample


def _sample(index: int, y: float = 600.0) -> MLTrainingSample:
    sequence = tuple(
        tuple(0.01 * (index + step + 1) for _ in SEQUENCE_FEATURE_NAMES)
        for step in range(60)
    )
    graph = (
        tuple(0.02 * (index + 1) for _ in GRAPH_FEATURE_NAMES),
        tuple(0.03 * (index + 1) for _ in GRAPH_FEATURE_NAMES),
    )
    context = tuple(0.04 * (index + 1) for _ in CONTEXT_FEATURE_NAMES)
    inputs = MLModelInputs(
        x_seq=sequence,
        x_graph=graph,
        edge_index=((0, 1), (1, 0)),
        operational_edge_index=((0,), (1,)),
        operational_edge_types=("ROUTE_SUCCESSOR",),
        current_node_index=index % 2,
        route_mask=(1.0, 1.0),
        x_context=context,
        node_ids=("UP-BLK-01", "UP-BLK-02"),
    )
    return MLTrainingSample(
        inputs=inputs,
        y=y,
        run_id=f"run-{index}",
        scenario_id=f"scenario-{index}",
        random_seed=index,
        train_id="TRAIN-1",
        sim_time_s=float(index),
    )


def test_hybrid_collation_matches_full_model_contract():
    batch = collate_hybrid_training_samples([_sample(0), _sample(1)])

    assert tuple(batch.x_seq.shape) == (2, 60, len(SEQUENCE_FEATURE_NAMES))
    assert tuple(batch.x_graph.shape) == (2, 2, len(GRAPH_FEATURE_NAMES))
    assert tuple(batch.edge_index.shape) == (2, 2)
    assert tuple(batch.current_node_index.shape) == (2,)
    assert tuple(batch.route_mask.shape) == (2, 2)
    assert tuple(batch.x_context.shape) == (2, len(CONTEXT_FEATURE_NAMES))
    assert tuple(batch.y.shape) == (2,)


def test_hybrid_forward_returns_one_non_negative_eta_per_sample():
    torch.manual_seed(17)
    model = HybridLSTMGraphSAGEETARegressor(
        lstm_hidden_size=16,
        lstm_layers=2,
        gnn_hidden_size=16,
        gnn_layers=3,
        dropout=0.0,
    )
    batch = collate_hybrid_training_samples([_sample(0), _sample(1), _sample(2)])

    prediction = model(
        batch.x_seq,
        batch.x_graph,
        batch.edge_index,
        batch.current_node_index,
        batch.route_mask,
        batch.x_context,
    )

    assert tuple(prediction.shape) == (3,)
    assert torch.all(torch.isfinite(prediction))
    assert torch.all(prediction >= 0)


def test_hybrid_training_updates_temporal_and_graph_branches():
    torch.manual_seed(23)
    model = HybridLSTMGraphSAGEETARegressor(
        lstm_hidden_size=12,
        lstm_layers=1,
        gnn_hidden_size=12,
        gnn_layers=2,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = [_sample(i, y=300.0 + 30.0 * i) for i in range(4)]

    lstm_before = model.lstm.weight_ih_l0.detach().clone()
    gnn_before = model.graph_encoder.layers[0].self_linear.weight.detach().clone()
    loss = train_hybrid_epoch(model, samples, optimizer, batch_size=2, shuffle_seed=5)

    assert math.isfinite(loss)
    assert loss >= 0.0
    assert not torch.equal(lstm_before, model.lstm.weight_ih_l0.detach())
    assert not torch.equal(gnn_before, model.graph_encoder.layers[0].self_linear.weight.detach())


def test_hybrid_predictor_adapter_returns_python_eta():
    torch.manual_seed(29)
    model = HybridLSTMGraphSAGEETARegressor(
        lstm_hidden_size=8,
        lstm_layers=1,
        gnn_hidden_size=8,
        gnn_layers=2,
        dropout=0.0,
    )

    prediction = TorchHybridETAPredictor(model).predict_seconds(_sample(0).inputs)

    assert isinstance(prediction, float)
    assert math.isfinite(prediction)
    assert prediction >= 0.0


def test_hybrid_collation_rejects_mixed_topology():
    first = _sample(0)
    second = _sample(1)
    changed_inputs = MLModelInputs(
        x_seq=second.inputs.x_seq,
        x_graph=second.inputs.x_graph,
        edge_index=((0,), (1,)),
        operational_edge_index=second.inputs.operational_edge_index,
        operational_edge_types=second.inputs.operational_edge_types,
        current_node_index=second.inputs.current_node_index,
        route_mask=second.inputs.route_mask,
        x_context=second.inputs.x_context,
        node_ids=second.inputs.node_ids,
    )
    changed = MLTrainingSample(
        inputs=changed_inputs,
        y=second.y,
        run_id=second.run_id,
        scenario_id=second.scenario_id,
        random_seed=second.random_seed,
        train_id=second.train_id,
        sim_time_s=second.sim_time_s,
    )

    try:
        collate_hybrid_training_samples([first, changed])
    except ValueError as exc:
        assert "topology" in str(exc)
    else:
        raise AssertionError("Expected mixed hybrid topology to be rejected")
