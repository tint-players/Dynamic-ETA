from __future__ import annotations

import pytest

from backend.session import SessionManager, sessions
from simulator.config_loader import simulation_config_from_dict
from simulator.models import TemporarySpeedRestriction
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from simulator.track_aware_exporters import TrackAwareParquetTelemetryExporter, TrackBlockVisitExporter
from simulator.track_blocks import enumerate_track_blocks, get_track_block, locate_track_block, track_block_id
from simulator.track_block_validation import validate_track_block_config


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _route():
    return _config().route


def test_delhi_agra_expands_to_sixteen_track_blocks_in_stable_order():
    sections = enumerate_track_blocks(_route())

    assert len(sections) == 16
    assert [section.track_block_id for section in sections[:8]] == [f"UP-BLK-{i:02d}" for i in range(1, 9)]
    assert [section.track_block_id for section in sections[8:]] == [f"DOWN-BLK-{i:02d}" for i in range(1, 9)]
    assert len({section.track_block_id for section in sections}) == 16


def test_up_and_down_sections_share_geometry_but_have_distinct_operational_identity():
    route = _route()
    up = get_track_block(route, "TRACK-UP", "BLK-05")
    down = get_track_block(route, "TRACK-DOWN", "BLK-05")

    assert up.track_block_id == "UP-BLK-05"
    assert down.track_block_id == "DOWN-BLK-05"
    assert up.geometry is down.geometry
    assert up.route_start_m == down.route_start_m
    assert up.route_end_m == down.route_end_m
    assert up.track_id != down.track_id


def test_locate_track_block_preserves_shared_route_geometry_and_selected_track():
    route = _route()
    start = route.block_start_distance_m("BLK-03")

    section, local = locate_track_block(route, "TRACK-DOWN", start + 350.0)

    assert section.track_block_id == "DOWN-BLK-03"
    assert section.block_id == "BLK-03"
    assert local == pytest.approx(350.0)


def test_unknown_track_or_block_is_rejected():
    route = _route()

    with pytest.raises(KeyError, match="Unknown track_id"):
        get_track_block(route, "TRACK-NOT-REAL", "BLK-01")
    with pytest.raises(KeyError, match="Unknown block_id"):
        get_track_block(route, "TRACK-UP", "BLK-99")


def test_track_block_id_is_stable_and_validates_empty_components():
    assert track_block_id("TRACK-UP", "BLK-07") == "UP-BLK-07"
    assert track_block_id("TRACK-DOWN", "BLK-07") == "DOWN-BLK-07"

    with pytest.raises(ValueError, match="track_id"):
        track_block_id("", "BLK-01")
    with pytest.raises(ValueError, match="block_id"):
        track_block_id("TRACK-UP", "")


def test_multitrack_restriction_requires_explicit_track():
    config = _config()
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="AMBIGUOUS",
            block_id="BLK-02",
            start_position_m=100.0,
            end_position_m=200.0,
            speed_limit_kmh=40.0,
        )
    ]

    with pytest.raises(ValueError, match="must specify track_id"):
        validate_track_block_config(config)


def test_track_block_validator_accepts_track_specific_restriction_and_shared_weather():
    config = _config()
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="UP-ONLY",
            block_id="BLK-02",
            track_id="TRACK-UP",
            start_position_m=100.0,
            end_position_m=200.0,
            speed_limit_kmh=40.0,
        )
    ]

    assert validate_track_block_config(config) is config
    assert config.environment.weather[1].block_id == "BLK-02"


def test_dict_loader_rejects_ambiguous_multitrack_restriction():
    raw = _config().model_dump(mode="python")
    raw["environment"]["temporary_speed_restrictions"] = [{
        "restriction_id": "AMBIGUOUS-LOAD",
        "block_id": "BLK-03",
        "start_position_m": 100.0,
        "end_position_m": 250.0,
        "speed_limit_kmh": 45.0,
        "start_time_s": 0.0,
        "end_time_s": None,
    }]

    with pytest.raises(ValueError, match="must specify track_id"):
        simulation_config_from_dict(raw)


def test_tick_export_contains_canonical_track_block_id():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="v12-export")
    train = next(item for item in engine.trains if item.train.train_id == "TRAIN-12002")
    block_start = config.route.block_start_distance_m("BLK-07")
    train.route_position_m = block_start + 2100.0
    train.current_track_id = "TRACK-UP"

    frame = engine.snapshot_train(train)
    telemetry = TrackAwareParquetTelemetryExporter(config).to_dataframe([frame])

    assert telemetry.iloc[0]["block_id"] == "BLK-07"
    assert telemetry.iloc[0]["track_block_id"] == "UP-BLK-07"


def test_crossover_track_change_splits_logical_block_into_two_track_block_visits():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="v12-visits")
    train = next(item for item in engine.trains if item.train.train_id == "TRAIN-12002")
    block_start = config.route.block_start_distance_m("BLK-07")

    train.route_position_m = block_start + 2150.0
    train.current_track_id = "TRACK-UP"
    up_frame = engine.snapshot_train(train).model_copy(update={"tick": 1, "sim_time_s": 1.0})

    train.route_position_m = block_start + 2250.0
    train.current_track_id = "TRACK-DOWN"
    down_frame = engine.snapshot_train(train).model_copy(update={"tick": 2, "sim_time_s": 2.0})

    telemetry = TrackAwareParquetTelemetryExporter(config).to_dataframe([up_frame, down_frame])
    visits = TrackBlockVisitExporter(config).to_dataframe(telemetry)

    assert visits["track_block_id"].tolist() == ["UP-BLK-07", "DOWN-BLK-07"]
    assert visits["block_id"].tolist() == ["BLK-07", "BLK-07"]
    assert visits["track_id_at_entry"].tolist() == ["TRACK-UP", "TRACK-DOWN"]
    assert visits["track_id_at_exit"].tolist() == ["TRACK-UP", "TRACK-DOWN"]


def test_multitrack_session_always_uses_restrictive_network_engine():
    session = sessions.create("delhi_agra_corridor.yaml")
    assert isinstance(session.engine, NetworkSimulationEngineV4Restrictive)


def test_manual_restriction_requires_track_and_exposes_track_block_id():
    session = sessions.create("delhi_agra_corridor.yaml")

    with pytest.raises(ValueError, match="track_id is required"):
        session.inject_speed_restriction(
            kind="tsr",
            block_id="BLK-03",
            track_id=None,
            start_position_m=100.0,
            end_position_m=300.0,
            speed_limit_kmh=40.0,
            duration_s=60.0,
        )

    session.inject_speed_restriction(
        kind="tsr",
        block_id="BLK-03",
        track_id="TRACK-DOWN",
        start_position_m=100.0,
        end_position_m=300.0,
        speed_limit_kmh=40.0,
        duration_s=60.0,
    )
    active = session.active_constraints()["tsr"][-1]

    assert active["track_id"] == "TRACK-DOWN"
    assert active["block_id"] == "BLK-03"
    assert active["track_block_id"] == "DOWN-BLK-03"
