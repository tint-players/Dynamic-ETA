from __future__ import annotations

import math

import torch

from simulator.ml_features import CONTEXT_FEATURE_NAMES, GRAPH_FEATURE_NAMES
from simulator.ml_gnn import (
    GraphSAGEEncoder,
    GraphSAGEETARegressor,
    MeanGraphSAGELayer,
    TorchGraphSAGEETAPredictor,
    collate_gnn_training_samples,
    train_gnn_epoch,
)
from simulator.ml_samples import MLModelInputs, MLTrainingSample


NODE_IDS = ("UP-BLK-01", "UP-BLK-02", "UP-BLK-03")
EDGE_INDEX = ((0, 1, 1, 2), (1, 0, 2, 1))


def _sample(index: int, y: float = 600.0) -> MLTrainingSample:
    graph = tuple(
        tuple(0.01 * (index + node + 1) for _ in GRAPH_FEATURE_NAMES)
        for node in range(len(NODE_IDS))
    )
    context = tuple(0.02 * (index + 1) for _ in CONTEXT_FEATURE_NAMES)
    inputs = MLModelInputs(
        x_seq=((0.0,),),
        x_graph=graph,
        edge_index=EDGE_INDEX,
        operational_edge_index=((0, 1), (1, 2)),
        operational_edge_types=("ROUTE_SUCCESSOR", "ROUTE_SUCCESSOR"),
        current_node_index=index % len(NODE_IDS),
        route_mask=(1.0, 1.0, 1.0),
        x_context=context,
        node_ids=NODE_IDS,
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


def test_gnn_collation_matches_model_contract_shapes():
    batch = collate_gnn_training_samples([_sample(0), _sample(1)])

    assert tuple(batch.x_graph.shape) == (2, 3, len(GRAPH_FEATURE_NAMES))
    assert tuple(batch.edge_index.shape) == (2, 4)
    assert tuple(batch.current_node_index.shape) == (2,)
    assert tuple(batch.route_mask.shape) == (2, 3)
    assert tuple(batch.x_context.shape) == (2, len(CONTEXT_FEATURE_NAMES))
    assert tuple(batch.y.shape) == (2,)
    assert batch.batch_size == 2


def test_mean_graphsage_aggregates_neighbour_mean():
    layer = MeanGraphSAGELayer(1, 1)
    with torch.no_grad():
        layer.self_linear.weight.fill_(0.0)
        layer.self_linear.bias.fill_(0.0)
        layer.neighbour_linear.weight.fill_(1.0)

    x = torch.tensor([[[1.0], [2.0], [5.0]]])
    edge_index = torch.tensor(((0, 2), (1, 1)), dtype=torch.long)
    output = layer(x, edge_index)

    assert torch.allclose(output[0, 1], torch.tensor([3.0]))
    assert torch.allclose(output[0, 0], torch.tensor([0.0]))
    assert torch.allclose(output[0, 2], torch.tensor([0.0]))


def test_graphsage_encoder_keeps_node_axis_and_hidden_size():
    torch.manual_seed(3)
    encoder = GraphSAGEEncoder(hidden_size=16, num_layers=3, dropout=0.0)
    batch = collate_gnn_training_samples([_sample(0), _sample(1)])

    encoded = encoder(batch.x_graph, batch.edge_index)

    assert tuple(encoded.shape) == (2, 3, 16)
    assert torch.all(encoded >= 0)


def test_gnn_forward_is_non_negative_and_one_value_per_sample():
    torch.manual_seed(7)
    model = GraphSAGEETARegressor(hidden_size=16, num_layers=3, dropout=0.0)
    batch = collate_gnn_training_samples([_sample(0), _sample(1), _sample(2)])

    prediction = model(
        batch.x_graph,
        batch.edge_index,
        batch.current_node_index,
        batch.route_mask,
        batch.x_context,
    )

    assert tuple(prediction.shape) == (3,)
    assert torch.all(prediction >= 0)


def test_gnn_single_training_epoch_updates_parameters():
    torch.manual_seed(11)
    model = GraphSAGEETARegressor(hidden_size=16, num_layers=2, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = [_sample(i, y=300.0 + 30.0 * i) for i in range(4)]

    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_gnn_epoch(model, samples, optimizer, batch_size=2, shuffle_seed=3)
    after = list(model.parameters())

    assert math.isfinite(loss)
    assert loss >= 0.0
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after))


def test_gnn_predictor_adapter_returns_python_non_negative_eta():
    torch.manual_seed(5)
    model = GraphSAGEETARegressor(hidden_size=8, num_layers=2, dropout=0.0)
    sample = _sample(0)

    prediction = TorchGraphSAGEETAPredictor(model).predict_seconds(sample.inputs)

    assert isinstance(prediction, float)
    assert math.isfinite(prediction)
    assert prediction >= 0.0


def test_gnn_collation_rejects_mixed_topologies():
    first = _sample(0)
    second = _sample(1)
    altered_inputs = MLModelInputs(
        x_seq=second.inputs.x_seq,
        x_graph=second.inputs.x_graph,
        edge_index=((0,), (2,)),
        operational_edge_index=second.inputs.operational_edge_index,
        operational_edge_types=second.inputs.operational_edge_types,
        current_node_index=second.inputs.current_node_index,
        route_mask=second.inputs.route_mask,
        x_context=second.inputs.x_context,
        node_ids=second.inputs.node_ids,
    )
    altered = MLTrainingSample(
        inputs=altered_inputs,
        y=second.y,
        run_id=second.run_id,
        scenario_id=second.scenario_id,
        random_seed=second.random_seed,
        train_id=second.train_id,
        sim_time_s=second.sim_time_s,
    )

    try:
        collate_gnn_training_samples([first, altered])
    except ValueError as exc:
        assert "same message-passing topology" in str(exc)
    else:
        raise AssertionError("Expected mixed-topology GNN batch to be rejected")
