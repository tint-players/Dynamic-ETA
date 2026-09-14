from __future__ import annotations

import pytest

from backend.session import SessionManager
from simulator.ml_features import CONTEXT_FEATURE_NAMES, GRAPH_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES, SEQUENCE_WINDOW_STEPS
from simulator.ml_samples import MLSampleBuilder
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from simulator.track_blocks import track_block_id


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _history_and_current(ticks: int = 3):
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-samples")
    history = engine.snapshot_all()
    for _ in range(ticks):
        history.extend(engine.tick())
    return config, history, engine.snapshot_all()


def _record(frame):
    row = frame.model_dump(mode="json")
    row["track_block_id"] = track_block_id(frame.track_id, frame.current_block_id)
    return row


def test_live_sample_combines_feature_topology_and_route_mask_contract():
    config, history, current = _history_and_current()
    target_train_id = config.train.train_id
    builder = MLSampleBuilder(config)

    sample = builder.build_inputs(
        history=history,
        current_frames=current,
        target_train_id=target_train_id,
    )

    assert len(sample.x_seq) == SEQUENCE_WINDOW_STEPS
    assert len(sample.x_seq[0]) == len(SEQUENCE_FEATURE_NAMES)
    assert len(sample.x_graph) == 16
    assert len(sample.x_graph[0]) == len(GRAPH_FEATURE_NAMES)
    assert len(sample.x_context) == len(CONTEXT_FEATURE_NAMES)
    assert len(sample.route_mask) == 16
    assert len(sample.edge_index[0]) == len(sample.edge_index[1]) == 30
    assert len(sample.operational_edge_index[0]) == len(sample.operational_edge_index[1]) == 15
    assert len(sample.operational_edge_types) == 15
    assert sample.node_ids[sample.current_node_index] == "UP-BLK-01"


def test_training_wrapper_attaches_label_and_provenance_without_changing_inputs():
    config, history, current = _history_and_current(ticks=2)
    target_train_id = config.train.train_id
    builder = MLSampleBuilder(config)

    live_inputs = builder.build_inputs(
        history=history,
        current_frames=current,
        target_train_id=target_train_id,
    )

    labelled_history = [
        frame.model_copy(update={"run_id": "run-sample-001", "random_seed": 42})
        for frame in history
    ]
    labelled_current = [
        frame.model_copy(update={
            "run_id": "run-sample-001",
            "random_seed": 42,
            "actual_remaining_time_s": 1234.0 if frame.train_id == target_train_id else 999.0,
            "actual_arrival_simulation_s": 2000.0,
            "total_journey_time_s": 2000.0,
        })
        for frame in current
    ]

    training = builder.build_training_sample(
        history=labelled_history,
        current_frames=labelled_current,
        target_train_id=target_train_id,
    )

    assert training.inputs == live_inputs
    assert training.y == 1234.0
    assert training.run_id == "run-sample-001"
    assert training.random_seed == 42
    assert training.train_id == target_train_id
    assert training.sim_time_s == current[0].sim_time_s


def test_training_sample_rejects_unlabelled_live_frame():
    config, history, current = _history_and_current(ticks=1)
    builder = MLSampleBuilder(config)

    with pytest.raises(ValueError, match="missing run_id"):
        builder.build_training_sample(
            history=history,
            current_frames=current,
            target_train_id=config.train.train_id,
        )


def test_offline_records_and_live_frames_produce_identical_inference_inputs():
    config, history, current = _history_and_current(ticks=2)
    builder = MLSampleBuilder(config)
    target_train_id = config.train.train_id

    live = builder.build_inputs(
        history=history,
        current_frames=current,
        target_train_id=target_train_id,
    )
    offline = builder.build_inputs_from_records(
        history_records=[_record(frame) for frame in history],
        current_records=[_record(frame) for frame in current],
        target_train_id=target_train_id,
    )

    assert offline == live


def test_offline_training_records_preserve_target_and_run_identity():
    config, history, current = _history_and_current(ticks=1)
    target_train_id = config.train.train_id
    builder = MLSampleBuilder(config)

    labelled_history = [
        frame.model_copy(update={"run_id": "run-record-001", "random_seed": 314})
        for frame in history
    ]
    labelled_current = [
        frame.model_copy(update={
            "run_id": "run-record-001",
            "random_seed": 314,
            "actual_remaining_time_s": 777.0 if frame.train_id == target_train_id else 888.0,
            "actual_arrival_simulation_s": 1000.0,
            "total_journey_time_s": 1000.0,
        })
        for frame in current
    ]

    sample = builder.build_training_sample_from_records(
        history_records=[_record(frame) for frame in labelled_history],
        current_records=[_record(frame) for frame in labelled_current],
        target_train_id=target_train_id,
    )

    assert sample.y == 777.0
    assert sample.run_id == "run-record-001"
    assert sample.random_seed == 314
