from __future__ import annotations

from dataclasses import replace

import pytest

from backend.session import SessionManager
from simulator.ml_baselines import (
    CurrentSpeedETABaseline,
    EffectiveCeilingETABaseline,
    HistoricalMeanSpeedETABaseline,
    evaluate_eta_predictor,
)
from simulator.ml_samples import MLModelInputs, MLTrainingSample


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _inputs(*, speed_norm: float, distance_ratio: float, ceiling_norm: float = 0.5) -> MLModelInputs:
    from simulator.ml_features import SEQUENCE_FEATURE_NAMES

    row = [0.0] * len(SEQUENCE_FEATURE_NAMES)
    row[SEQUENCE_FEATURE_NAMES.index("history_present")] = 1.0
    row[SEQUENCE_FEATURE_NAMES.index("speed_norm")] = speed_norm
    row[SEQUENCE_FEATURE_NAMES.index("distance_to_destination_ratio")] = distance_ratio
    row[SEQUENCE_FEATURE_NAMES.index("effective_speed_ceiling_norm")] = ceiling_norm
    sequence = tuple(tuple(row) for _ in range(60))
    return MLModelInputs(
        x_seq=sequence,
        x_graph=((0.0,),),
        edge_index=((), ()),
        operational_edge_index=((), ()),
        operational_edge_types=(),
        current_node_index=0,
        route_mask=(1.0,),
        x_context=(0.0,),
        node_ids=("UP-BLK-01",),
    )


def _sample(y: float, *, speed_norm: float = 0.5, distance_ratio: float = 0.5) -> MLTrainingSample:
    return MLTrainingSample(
        inputs=_inputs(speed_norm=speed_norm, distance_ratio=distance_ratio),
        y=y,
        run_id="run-baseline",
        scenario_id="baseline",
        random_seed=42,
        train_id="TRAIN-1",
        sim_time_s=100.0,
    )


def test_current_speed_baseline_uses_only_live_safe_inputs():
    config = _config()
    baseline = CurrentSpeedETABaseline.from_config(config, speed_floor_kmh=5.0)
    inputs = _inputs(speed_norm=0.5, distance_ratio=0.25)

    prediction = baseline.predict_seconds(inputs)

    expected_distance_m = 0.25 * config.route.total_length_m
    expected_speed_ms = 100.0 / 3.6
    assert prediction == pytest.approx(expected_distance_m / expected_speed_ms)


def test_historical_mean_speed_baseline_ignores_zero_padding():
    config = _config()
    baseline = HistoricalMeanSpeedETABaseline.from_config(config, speed_floor_kmh=5.0)
    inputs = _inputs(speed_norm=0.5, distance_ratio=0.2)
    rows = [list(row) for row in inputs.x_seq]
    from simulator.ml_features import SEQUENCE_FEATURE_NAMES

    history_idx = SEQUENCE_FEATURE_NAMES.index("history_present")
    speed_idx = SEQUENCE_FEATURE_NAMES.index("speed_norm")
    for row in rows[:-2]:
        row[history_idx] = 0.0
        row[speed_idx] = 0.0
    rows[-2][speed_idx] = 0.25  # 50 km/h
    rows[-1][speed_idx] = 0.50  # 100 km/h
    inputs = replace(inputs, x_seq=tuple(tuple(row) for row in rows))

    prediction = baseline.predict_seconds(inputs)

    expected_distance_m = 0.2 * config.route.total_length_m
    assert prediction == pytest.approx(expected_distance_m / (75.0 / 3.6))


def test_effective_ceiling_baseline_uses_current_ceiling():
    config = _config()
    baseline = EffectiveCeilingETABaseline.from_config(config)
    inputs = _inputs(speed_norm=0.1, distance_ratio=0.1, ceiling_norm=0.6)

    prediction = baseline.predict_seconds(inputs)

    expected_distance_m = 0.1 * config.route.total_length_m
    assert prediction == pytest.approx(expected_distance_m / (120.0 / 3.6))


def test_evaluation_reports_global_and_eta_bucket_metrics():
    class FixedPredictor:
        def predict_seconds(self, inputs):
            return 100.0

    samples = [
        _sample(60.0),
        _sample(600.0),
        _sample(1200.0),
        _sample(2400.0),
    ]
    metrics = evaluate_eta_predictor(FixedPredictor(), samples)

    assert metrics.sample_count == 4
    assert metrics.mae_s == pytest.approx((40.0 + 500.0 + 1100.0 + 2300.0) / 4)
    assert metrics.median_absolute_error_s == pytest.approx(800.0)
    assert len(metrics.buckets) == 4
    assert [bucket.sample_count for bucket in metrics.buckets] == [1, 1, 1, 1]


def test_evaluation_rejects_invalid_prediction():
    class InvalidPredictor:
        def predict_seconds(self, inputs):
            return -1.0

    with pytest.raises(ValueError, match="finite non-negative"):
        evaluate_eta_predictor(InvalidPredictor(), [_sample(60.0)])
