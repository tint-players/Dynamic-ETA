from __future__ import annotations

from backend.session import SessionManager
from simulator.models import MaintenanceRestriction, MaintenanceType
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def _isolated_primary_engine():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)
    config.environment.maintenance_restrictions = [
        MaintenanceRestriction(
            restriction_id="MAINT-CLOSE-01",
            block_id="BLK-02",
            start_position_m=600.0,
            end_position_m=900.0,
            speed_limit_kmh=20.0,
            maintenance_type=MaintenanceType.FULL_CLOSURE,
            start_time_s=100.0,
            end_time_s=200.0,
        )
    ]
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="phase2-v10-maintenance")
    primary = engine.trains[0]
    for train in engine.trains[1:]:
        train.completed = True
    return config, engine, primary


def test_full_closure_target_is_active_only_during_maintenance_window():
    config, engine, train = _isolated_primary_engine()
    block_start = config.route.block_start_distance_m("BLK-02")
    expected_stop = block_start + 600.0 - engine.MAINTENANCE_STOP_MARGIN_M
    train.route_position_m = expected_stop - 300.0
    train.source_m = 0.0
    train.destination_m = config.route.total_length_m

    engine.sim_time_s = 99.0
    assert not any(target.reason.startswith("MAINTENANCE_CLOSURE:") for target in engine._targets(train))

    engine.sim_time_s = 100.0
    target = next(target for target in engine._targets(train) if target.reason == "MAINTENANCE_CLOSURE:MAINT-CLOSE-01")
    assert target.hard_stop is True
    assert target.speed_kmh == 0.0
    assert target.position_m == expected_stop

    engine.sim_time_s = 200.0
    assert not any(target.reason.startswith("MAINTENANCE_CLOSURE:") for target in engine._targets(train))


def test_full_closure_holds_train_before_zone_and_releases_after_end():
    config, engine, train = _isolated_primary_engine()
    block_start = config.route.block_start_distance_m("BLK-02")
    expected_stop = block_start + 600.0 - engine.MAINTENANCE_STOP_MARGIN_M
    train.route_position_m = expected_stop - 1.0
    train.source_m = 0.0
    train.destination_m = config.route.total_length_m
    train.speed_kmh = 20.0

    engine.sim_time_s = 150.0
    action, reason = engine._advance(train, 1.0)
    assert action == "STOP"
    assert reason == "MAINTENANCE_CLOSURE:MAINT-CLOSE-01"
    assert train.speed_kmh == 0.0
    assert train.route_position_m < block_start + 600.0

    held_position = train.route_position_m
    action, reason = engine._advance(train, 1.0)
    assert train.speed_kmh == 0.0
    assert train.route_position_m <= held_position + 1e-6

    engine.sim_time_s = 200.0
    action, reason = engine._advance(train, 1.0)
    assert action in {"ACCELERATE", "MAINTAIN"}
    assert not reason.startswith("MAINTENANCE_CLOSURE:")
