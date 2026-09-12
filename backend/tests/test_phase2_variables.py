from __future__ import annotations

import pytest

from backend.session import SessionManager
from simulator.exporters import BlockVisitExporter, ParquetTelemetryExporter
from simulator.models import (
    MaintenanceRestriction,
    SignalAspect,
    TemporarySpeedRestriction,
    WeatherCondition,
    WeatherTimelineEntry,
)
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def test_phase2_baseline_is_clear_and_has_no_manual_speed_constraints():
    config = _config()
    assert config.environment.temporary_speed_restrictions == []
    assert config.environment.maintenance_restrictions == []
    assert all(
        len(schedule.timeline) == 1
        and schedule.timeline[0].condition == WeatherCondition.CLEAR
        and schedule.timeline[0].visibility_m == 10000
        for schedule in config.environment.weather
    )


def test_manual_weather_timeline_is_applied_by_simulation_time():
    config = _config()
    blk01 = next(item for item in config.environment.weather if item.block_id == "BLK-01")
    blk01.timeline = [
        WeatherTimelineEntry(start_time_s=0, condition=WeatherCondition.CLEAR, visibility_m=10000),
        WeatherTimelineEntry(start_time_s=30, condition=WeatherCondition.HEAVY_RAIN, visibility_m=1200),
    ]
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="weather-manual")
    engine.sim_time_s = 29
    assert engine._weather_at("BLK-01") == (WeatherCondition.CLEAR, 10000)
    engine.sim_time_s = 30
    assert engine._weather_at("BLK-01") == (WeatherCondition.HEAVY_RAIN, 1200)


def test_manual_tsr_and_maintenance_are_speed_limits_only():
    config = _config()
    config.environment.temporary_speed_restrictions = [TemporarySpeedRestriction(
        restriction_id="TSR-TEST", block_id="BLK-01", start_position_m=100,
        end_position_m=900, speed_limit_kmh=70, start_time_s=0, end_time_s=120,
    )]
    config.environment.maintenance_restrictions = [MaintenanceRestriction(
        restriction_id="MAINT-TEST", block_id="BLK-01", start_position_m=200,
        end_position_m=800, speed_limit_kmh=45, start_time_s=0, end_time_s=120,
    )]
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="manual-limits")
    train = engine.trains[0]
    train.route_position_m = 500
    engine.sim_time_s = 60
    ceiling, _, _ = engine._current_ceiling(train)
    assert ceiling == 45
    engine.sim_time_s = 121
    ceiling_after, _, _ = engine._current_ceiling(train)
    assert ceiling_after == 110


def test_manual_signal_override_expires_back_to_dynamic_signalling():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="manual-signal")
    signal = next(item for item in config.signals if item.signal_id == "UP-02")
    engine.set_manual_signal_override("UP-02", SignalAspect.RED, 20)
    assert engine.signal_aspect(signal) == SignalAspect.RED
    engine.sim_time_s = 19
    assert engine.signal_aspect(signal) == SignalAspect.RED
    engine.sim_time_s = 20
    assert "UP-02" not in engine.manual_signal_overrides()
    assert engine.signal_aspect(signal) == super(NetworkSimulationEngineV4Restrictive, engine).signal_aspect(signal)


def test_manual_signal_injection_rejects_green_and_reset_returns_selected_signal_to_automatic():
    session = SessionManager().create("delhi_agra_corridor.yaml")
    with pytest.raises(ValueError, match="only supports RED or YELLOW"):
        session.inject_signal("UP-04", SignalAspect.GREEN, 60)

    session.inject_signal("UP-04", SignalAspect.YELLOW, 120)
    session.inject_signal("UP-05", SignalAspect.RED, 120)
    assert session.engine.manual_signal_overrides()["UP-04"] == SignalAspect.YELLOW
    assert session.engine.manual_signal_overrides()["UP-05"] == SignalAspect.RED

    session.reset_signal("UP-04")
    overrides = session.engine.manual_signal_overrides()
    assert "UP-04" not in overrides
    assert overrides["UP-05"] == SignalAspect.RED
    assert all(item["signal_id"] != "UP-04" for item in session.active_constraints()["signals"])


def test_occupied_platform_adds_station_entry_hold_without_replacing_other_safety_targets():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="station-occupancy")
    approaching = next(item for item in engine.trains if item.train.train_id == "TRAIN-12002")
    occupant = next(item for item in engine.trains if item.train.train_id == "TRAIN-CROSS-UP")
    for other in engine.trains:
        if other is not approaching and other is not occupant:
            other.completed = True
    block_start = config.route.block_start_distance_m("BLK-03")
    platform = next(platform for station in config.stations if station.station_id == "MATHURA" for platform in station.platforms if platform.track_id == "TRACK-UP")
    center = block_start + platform.position_in_block_m
    platform_entrance = center - platform.length_m / 2
    approaching.current_track_id = "TRACK-UP"
    approaching.route_position_m = platform_entrance - 150
    approaching.source_m = approaching.route_position_m
    approaching.destination_m = config.route.total_length_m
    approaching.served_stations.clear()
    occupant.current_track_id = "TRACK-UP"
    occupant.route_position_m = center + 50
    occupant.source_m = occupant.route_position_m
    occupant.destination_m = config.route.total_length_m
    occupant.completed = False
    occupant.departure_time_s = 0
    targets = engine._station_occupancy_targets(approaching)
    mathura = [target for target in targets if target.reason.startswith("STATION_OCCUPIED:MATHURA:")]
    assert len(mathura) == 1
    assert mathura[0].position_m == platform_entrance - engine.STATION_ENTRY_MARGIN_M
    assert mathura[0].hard_stop is True


def test_tickwise_enriched_export_and_block_visit_export_keep_separate_granularity():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="phase2-export")
    frames = engine.snapshot_all()
    for _ in range(5):
        frames.extend(engine.tick())
    telemetry = ParquetTelemetryExporter(config).to_dataframe(frames)
    assert len(telemetry) == len(frames)
    assert set(telemetry["sim_time_s"].unique()) == {0, 1, 2, 3, 4, 5}
    assert {"tsr_active", "maintenance_active", "station_platform_occupied_on_approach", "station_occupancy_wait_s", "train_ahead_present", "distance_to_train_ahead_m", "speed_gradient_30s"}.issubset(telemetry.columns)
    block_visits = BlockVisitExporter(config).to_dataframe(telemetry)
    assert not block_visits.empty
    assert {"entry_sim_time_s", "exit_sim_time_s", "actual_block_time_s", "tsr_exposure_s", "maintenance_exposure_s", "station_occupancy_wait_s", "traffic_hold_time_s"}.issubset(block_visits.columns)


def test_timed_weather_expires_to_clear_and_constraint_state_expires():
    session = SessionManager().create("delhi_agra_corridor.yaml")
    session.engine.sim_time_s = 40
    session.inject_weather("BLK-02", WeatherCondition.FOG, 1000, 30)
    weather = next(item for item in session.config.environment.weather if item.block_id == "BLK-02")
    assert session.engine._weather_at("BLK-02") == (WeatherCondition.FOG, 1000)
    assert session.active_constraints()["weather"][0]["end_time_s"] == 70
    session.engine.sim_time_s = 70
    assert session.engine._weather_at("BLK-02") == (WeatherCondition.CLEAR, 10000)
    assert session.active_constraints()["weather"] == []
    assert weather.timeline[-1].condition == WeatherCondition.CLEAR


def test_reset_constraints_preserves_simulation_position_and_clears_manual_state():
    session = SessionManager().create("delhi_agra_corridor.yaml")
    session.engine.sim_time_s = 40
    session.inject_weather("BLK-02", WeatherCondition.FOG, 1000, 300)
    session.inject_speed_restriction("tsr", "BLK-03", 100, 300, 65, 180)
    session.inject_speed_restriction("maintenance", "BLK-04", 200, 400, 40, 240)
    session.inject_signal("UP-04", SignalAspect.RED, 60)
    session.reset_constraints()
    assert session.engine.sim_time_s == 40
    assert session.config.environment.temporary_speed_restrictions == []
    assert session.config.environment.maintenance_restrictions == []
    assert session.engine.manual_signal_overrides() == {}
    assert session.active_constraints()["weather"] == []
    weather = next(item for item in session.config.environment.weather if item.block_id == "BLK-02")
    assert len(weather.timeline) == 1
    assert weather.timeline[0].condition == WeatherCondition.CLEAR


def test_session_live_injection_mutates_current_run_and_reset_restores_yaml_baseline():
    session = SessionManager().create("delhi_agra_corridor.yaml")
    session.engine.sim_time_s = 40
    session.inject_weather("BLK-02", WeatherCondition.FOG, 900, 120)
    session.inject_speed_restriction("tsr", "BLK-03", 100, 700, 65, 180)
    session.inject_speed_restriction("maintenance", "BLK-04", 200, 900, 40, 240)
    session.inject_signal("UP-04", SignalAspect.RED, 60)
    assert len(session.config.environment.temporary_speed_restrictions) == 1
    assert len(session.config.environment.maintenance_restrictions) == 1
    assert session.engine.manual_signal_overrides()["UP-04"] == SignalAspect.RED
    session.reset()
    weather_after = next(item for item in session.config.environment.weather if item.block_id == "BLK-02")
    assert len(weather_after.timeline) == 1
    assert weather_after.timeline[0].condition == WeatherCondition.CLEAR
    assert session.config.environment.temporary_speed_restrictions == []
    assert session.config.environment.maintenance_restrictions == []
    assert session.engine.manual_signal_overrides() == {}
    assert session.engine.sim_time_s == 0
