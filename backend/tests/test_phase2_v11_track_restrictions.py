from __future__ import annotations

import pytest

from backend.session import SessionManager, sessions
from simulator.models import MaintenanceRestriction, MaintenanceType, TemporarySpeedRestriction
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def _engine():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="phase2-v11-track-restrictions")
    engine.sim_time_s = 100.0
    return config, engine


def test_track_specific_tsr_only_changes_selected_track_ceiling():
    config, engine = _engine()
    block_start = config.route.block_start_distance_m("BLK-02")
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="TSR-UP-ONLY",
            block_id="BLK-02",
            track_id="TRACK-UP",
            start_position_m=600.0,
            end_position_m=900.0,
            speed_limit_kmh=20.0,
            start_time_s=0.0,
            end_time_s=500.0,
        )
    ]

    up = next(train for train in engine.trains if train.current_track_id == "TRACK-UP")
    down = next(train for train in engine.trains if train.current_track_id == "TRACK-DOWN")
    for train in (up, down):
        train.route_position_m = block_start + 700.0

    up_ceiling = engine._current_ceiling(up)[0]
    down_ceiling = engine._current_ceiling(down)[0]

    assert up_ceiling == pytest.approx(20.0)
    assert down_ceiling > 20.0


def test_full_closure_only_creates_stop_target_on_selected_track():
    config, engine = _engine()
    block_start = config.route.block_start_distance_m("BLK-02")
    config.environment.maintenance_restrictions = [
        MaintenanceRestriction(
            restriction_id="MAINT-UP-ONLY",
            block_id="BLK-02",
            track_id="TRACK-UP",
            start_position_m=1000.0,
            end_position_m=1200.0,
            speed_limit_kmh=20.0,
            maintenance_type=MaintenanceType.FULL_CLOSURE,
            start_time_s=0.0,
            end_time_s=500.0,
        )
    ]

    up = next(train for train in engine.trains if train.current_track_id == "TRACK-UP")
    down = next(train for train in engine.trains if train.current_track_id == "TRACK-DOWN")
    for train in (up, down):
        train.source_m = 0.0
        train.destination_m = config.route.total_length_m
        train.route_position_m = block_start + 500.0

    up_targets = engine._maintenance_closure_targets(up)
    down_targets = engine._maintenance_closure_targets(down)

    assert [target.reason for target in up_targets] == ["MAINTENANCE_CLOSURE:MAINT-UP-ONLY"]
    assert down_targets == []


def test_manual_session_injection_records_selected_track():
    session = sessions.create("delhi_agra_corridor.yaml")
    message = session.inject_speed_restriction(
        kind="tsr",
        block_id="BLK-03",
        track_id="TRACK-DOWN",
        start_position_m=100.0,
        end_position_m=300.0,
        speed_limit_kmh=40.0,
        duration_s=60.0,
    )

    active = session.active_constraints()["tsr"]
    assert active[-1]["track_id"] == "TRACK-DOWN"
    assert "TRACK-DOWN BLK-03" in message


def test_unknown_restriction_track_is_rejected_by_config_validation():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="BAD-TRACK",
            block_id="BLK-02",
            track_id="TRACK-NOT-REAL",
            start_position_m=100.0,
            end_position_m=200.0,
            speed_limit_kmh=30.0,
        )
    ]

    with pytest.raises(ValueError, match="unknown track"):
        config.model_validate(config.model_dump())
