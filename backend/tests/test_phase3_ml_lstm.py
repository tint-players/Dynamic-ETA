from __future__ import annotations

import math

import torch

from simulator.ml_features import CONTEXT_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES
from simulator.ml_lstm import (
    LSTMETARegressor,
    TorchLSTMETAPredictor,
    collate_lstm_training_samples,
    huber_eta_loss,
    train_lstm_epoch,
)
from simulator.ml_samples import MLModelInputs, MLTrainingSample


def _sample(index: int, y: float = 600.0) -> MLTrainingSample:
    row = tuple(0.01 * (index + 1) for _ in SEQUENCE_FEATURE_NAMES)
    sequence = tuple(row for _ in range(60))
    context = tuple(0.02 * (index + 1) for _ in CONTEXT_FEATURE_NAMES)
    inputs = MLModelInputs(
        x_seq=sequence,
        x_graph=((0.0,),),
        edge_index=((), ()),
        operational_edge_index=((), ()),
        operational_edge_types=(),
        current_node_index=0,
        route_mask=(1.0,),
        x_context=context,
        node_ids=("UP-BLK-01",),
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


def test_lstm_collation_matches_model_contract_shapes():
    batch = collate_lstm_training_samples([_sample(0), _sample(1)])

    assert tuple(batch.x_seq.shape) == (2, 60, len(SEQUENCE_FEATURE_NAMES))
    assert tuple(batch.x_context.shape) == (2, len(CONTEXT_FEATURE_NAMES))
    assert tuple(batch.y.shape) == (2,)
    assert batch.batch_size == 2


def test_lstm_forward_is_non_negative_and_one_value_per_sample():
    torch.manual_seed(7)
    model = LSTMETARegressor(hidden_size=16, num_layers=2, dropout=0.0)
    batch = collate_lstm_training_samples([_sample(0), _sample(1), _sample(2)])

    prediction = model(batch.x_seq, batch.x_context)

    assert tuple(prediction.shape) == (3,)
    assert torch.all(prediction >= 0)


def test_huber_eta_loss_and_single_training_epoch_are_finite():
    torch.manual_seed(11)
    model = LSTMETARegressor(hidden_size=16, num_layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    samples = [_sample(i, y=300.0 + 30.0 * i) for i in range(4)]

    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_lstm_epoch(model, samples, optimizer, batch_size=2, shuffle_seed=3)
    after = list(model.parameters())

    assert math.isfinite(loss)
    assert loss >= 0.0
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after))

    prediction = torch.tensor([10.0, 20.0])
    target = torch.tensor([12.0, 25.0])
    assert float(huber_eta_loss(prediction, target)) >= 0.0


def test_lstm_predictor_adapter_returns_python_non_negative_eta():
    torch.manual_seed(5)
    model = LSTMETARegressor(hidden_size=8, num_layers=1, dropout=0.0)
    sample = _sample(0)

    prediction = TorchLSTMETAPredictor(model).predict_seconds(sample.inputs)

    assert isinstance(prediction, float)
    assert math.isfinite(prediction)
    assert prediction >= 0.0
