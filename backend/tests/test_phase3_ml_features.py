from __future__ import annotations

from backend.session import SessionManager
from simulator.ml_contract import ML_FORBIDDEN_LIVE_INPUT_FIELDS
from simulator.ml_features import (
    CONTEXT_FEATURE_NAMES,
    GRAPH_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SEQUENCE_WINDOW_STEPS,
    MLFeatureBuilder,
)
from simulator.models import TemporarySpeedRestriction
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from simulator.track_blocks import enumerate_track_blocks, track_block_id


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _short_history(engine: NetworkSimulationEngineV4Restrictive, ticks: int = 3):
    history = engine.snapshot_all()
    for _ in range(ticks):
        history.extend(engine.tick())
    return history, engine.snapshot_all()


def test_feature_builder_produces_fixed_60_step_sequence_and_16_track_block_graph():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-feature-shape")
    history, current = _short_history(engine, ticks=3)
    target_train_id = config.train.train_id

    builder = MLFeatureBuilder(config)
    sample = builder.build(history=history, current_frames=current, target_train_id=target_train_id)

    assert sample.sequence_shape == (SEQUENCE_WINDOW_STEPS, len(SEQUENCE_FEATURE_NAMES))
    assert sample.graph_shape == (16, len(GRAPH_FEATURE_NAMES))
    assert len(sample.context) == len(CONTEXT_FEATURE_NAMES)
    assert sample.node_ids == tuple(section.track_block_id for section in enumerate_track_blocks(config.route))

    present_index = SEQUENCE_FEATURE_NAMES.index("history_present")
    assert sum(row[present_index] for row in sample.sequence) == 4.0
    assert sample.sequence[-1][present_index] == 1.0


def test_current_node_index_uses_canonical_track_block_identity():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-current-node")
    current = engine.snapshot_all()
    target = next(frame for frame in current if frame.train_id == config.train.train_id)

    builder = MLFeatureBuilder(config)
    sample = builder.build(history=current, current_frames=current, target_train_id=target.train_id)

    assert sample.node_ids[sample.current_node_index] == track_block_id(target.track_id, target.current_block_id)


def test_post_run_labels_cannot_change_live_feature_tensors():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-no-leakage")
    history, current = _short_history(engine, ticks=2)
    target_train_id = config.train.train_id
    builder = MLFeatureBuilder(config)

    baseline = builder.build(history=history, current_frames=current, target_train_id=target_train_id)
    labelled_history = [
        frame.model_copy(update={
            "actual_remaining_time_s": 1234.0,
            "actual_arrival_simulation_s": 5678.0,
            "total_journey_time_s": 4321.0,
        })
        for frame in history
    ]
    labelled_current = [
        frame.model_copy(update={
            "actual_remaining_time_s": 1.0,
            "actual_arrival_simulation_s": 2.0,
            "total_journey_time_s": 3.0,
        })
        for frame in current
    ]
    labelled = builder.build(
        history=labelled_history,
        current_frames=labelled_current,
        target_train_id=target_train_id,
    )

    assert baseline == labelled
    all_feature_names = set(SEQUENCE_FEATURE_NAMES) | set(GRAPH_FEATURE_NAMES) | set(CONTEXT_FEATURE_NAMES)
    assert ML_FORBIDDEN_LIVE_INPUT_FIELDS.isdisjoint(all_feature_names)


def test_graph_restriction_state_is_track_specific():
    config = _config()
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="ML-UP-ONLY",
            block_id="BLK-05",
            track_id="TRACK-UP",
            start_position_m=0.0,
            end_position_m=config.route.blocks[config.route.block_index("BLK-05")].length_m,
            speed_limit_kmh=40.0,
            start_time_s=0.0,
        )
    ]
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-track-tsr")
    current = engine.snapshot_all()
    builder = MLFeatureBuilder(config)
    sample = builder.build(history=current, current_frames=current, target_train_id=config.train.train_id)

    tsr_index = GRAPH_FEATURE_NAMES.index("tsr_active")
    limit_index = GRAPH_FEATURE_NAMES.index("tsr_limit_norm")
    up = sample.graph[sample.node_ids.index("UP-BLK-05")]
    down = sample.graph[sample.node_ids.index("DOWN-BLK-05")]

    assert up[tsr_index] == 1.0
    assert up[limit_index] == 40.0 / 200.0
    assert down[tsr_index] == 0.0
    assert down[limit_index] == 0.0


def test_graph_occupancy_distinguishes_parallel_tracks_and_uses_train_body():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-track-occupancy")
    base = next(frame for frame in engine.snapshot_all() if frame.train_id == config.train.train_id)
    block = config.route.blocks[config.route.block_index("BLK-05")]
    route_position = config.route.block_start_distance_m("BLK-05") + min(100.0, block.length_m / 2)
    frame = base.model_copy(update={
        "track_id": "TRACK-UP",
        "current_block_id": "BLK-05",
        "position_in_block_m": min(100.0, block.length_m / 2),
        "route_position_m": route_position,
        "active": True,
        "completed": False,
    })

    builder = MLFeatureBuilder(config)
    sample = builder.build(history=[frame], current_frames=[frame], target_train_id=frame.train_id)
    occupied_index = GRAPH_FEATURE_NAMES.index("occupied")

    assert sample.graph[sample.node_ids.index("UP-BLK-05")][occupied_index] == 1.0
    assert sample.graph[sample.node_ids.index("DOWN-BLK-05")][occupied_index] == 0.0


def test_parquet_style_records_and_live_frames_share_exact_feature_path():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-offline-live-parity")
    history, current = _short_history(engine, ticks=2)
    target_train_id = config.train.train_id
    builder = MLFeatureBuilder(config)

    live = builder.build(history=history, current_frames=current, target_train_id=target_train_id)

    def record(frame):
        row = frame.model_dump(mode="json")
        row["track_block_id"] = track_block_id(frame.track_id, frame.current_block_id)
        return row

    offline = builder.build_from_records(
        history_records=[record(frame) for frame in history],
        current_records=[record(frame) for frame in current],
        target_train_id=target_train_id,
    )

    assert offline == live
