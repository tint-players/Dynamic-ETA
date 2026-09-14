from __future__ import annotations

import math

import torch

from backend.session import SessionManager
from simulator.ml_baselines import eta_metrics_from_predictions
from simulator.ml_experiment import ETAExperimentConfig, run_eta_experiment, save_winner_checkpoint
from simulator.ml_features import CONTEXT_FEATURE_NAMES, GRAPH_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES
from simulator.ml_samples import MLModelInputs, MLTrainingSample
from simulator.models import MaintenanceType
from simulator.scenario_generator import GeneratedScenarioRun, ScenarioGenerator, ScenarioGeneratorConfig


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _synthetic_sample(run_id: str, index: int) -> MLTrainingSample:
    sequence_row = [0.0 for _ in SEQUENCE_FEATURE_NAMES]
    sequence_row[SEQUENCE_FEATURE_NAMES.index("history_present")] = 1.0
    sequence_row[SEQUENCE_FEATURE_NAMES.index("speed_norm")] = 0.50
    sequence_row[SEQUENCE_FEATURE_NAMES.index("distance_to_destination_ratio")] = 0.50
    sequence_row[SEQUENCE_FEATURE_NAMES.index("effective_speed_ceiling_norm")] = 0.60
    sequence = tuple(tuple(sequence_row) for _ in range(60))
    graph_row = tuple(0.01 * (index + 1) for _ in GRAPH_FEATURE_NAMES)
    context = tuple(0.02 * (index + 1) for _ in CONTEXT_FEATURE_NAMES)
    inputs = MLModelInputs(
        x_seq=sequence,
        x_graph=(graph_row,),
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
        y=300.0 + 30.0 * index,
        run_id=run_id,
        scenario_id=f"scenario-{run_id}",
        random_seed=123,
        train_id="TRAIN-1",
        sim_time_s=float(index),
    )


def test_eta_metrics_from_batched_predictions_matches_expected_errors():
    metrics = eta_metrics_from_predictions([100.0, 210.0, 390.0], [120.0, 200.0, 400.0])

    assert metrics.sample_count == 3
    assert math.isclose(metrics.mae_s, 40.0 / 3.0)
    assert metrics.median_absolute_error_s == 10.0
    assert sum(bucket.sample_count for bucket in metrics.buckets) == 3


def test_randomized_maintenance_is_track_specific_and_finite():
    generator = ScenarioGenerator(
        _config(),
        ScenarioGeneratorConfig(
            n_scenarios=1,
            random_seed=99,
            baseline_probability=0.0,
            tsr_probability=0.0,
            crossing_closure_probability=0.0,
            signal_restriction_probability=0.0,
            maintenance_probability=1.0,
            maintenance_full_closure_probability=1.0,
        ),
    )

    generated = generator._make_scenario()
    assert len(generated.environment.maintenance_restrictions) == 1
    restriction = generated.environment.maintenance_restrictions[0]
    assert restriction.track_id in generated.route.track_ids
    assert restriction.maintenance_type == MaintenanceType.FULL_CLOSURE
    assert restriction.end_time_s is not None
    assert restriction.end_time_s > restriction.start_time_s


def test_full_experiment_selects_on_validation_and_saves_checkpoint(monkeypatch, tmp_path):
    config = _config()
    run_ids = ("run-a", "run-b", "run-c")
    runs = tuple(
        GeneratedScenarioRun(
            run_id=run_id,
            scenario_id=f"scenario-{run_id}",
            random_seed=123,
            config=config.model_copy(deep=True),
            frames=(),
        )
        for run_id in run_ids
    )
    samples = tuple(
        _synthetic_sample(run_id, index)
        for run_id in run_ids
        for index in range(3)
    )

    import simulator.ml_experiment as experiment_module

    monkeypatch.setattr(experiment_module, "build_experiment_samples", lambda _: samples)
    result = run_eta_experiment(
        runs,
        experiment=ETAExperimentConfig(
            epochs=1,
            batch_size=3,
            early_stopping_patience=1,
            min_validation_improvement_s=0.0,
            lstm_hidden_size=8,
            lstm_layers=1,
            gnn_hidden_size=8,
            gnn_layers=1,
            dropout=0.0,
            device="cpu",
        ),
    )

    assert {candidate.name for candidate in result.candidates} == {
        "current_speed",
        "historical_mean_speed",
        "effective_ceiling",
        "lstm",
        "graphsage",
        "hybrid",
    }
    assert result.winner_name in {candidate.name for candidate in result.candidates}
    assert result.winner_validation_metrics.sample_count == 3
    assert result.winner_test_metrics.sample_count == 3
    assert set(result.split.train_run_ids).isdisjoint(result.split.validation_run_ids)
    assert set(result.split.train_run_ids).isdisjoint(result.split.test_run_ids)
    assert set(result.split.validation_run_ids).isdisjoint(result.split.test_run_ids)

    checkpoint = tmp_path / "winner.pt"
    save_winner_checkpoint(result, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["model_name"] == result.winner_name
    assert payload["feature_contract"]["sequence_window_steps"] == 60
    assert tuple(payload["split_run_ids"]["test"]) == result.split.test_run_ids
