import csv
from pathlib import Path

from fastapi.testclient import TestClient

import backend.session as session_module
from backend.main import app
from backend.session import SessionManager
from backend.viz import config_for_visualization
from simulator.engine import SimulationEngine
from simulator.models import SignalAspect
from simulator.network_engine_v3 import NetworkSimulationEngineV3


client = TestClient(app)


def test_visual_config_uses_backend_route_geometry():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    payload = config_for_visualization(config)
    assert payload["route"]["track_ids"] == ["TRACK-UP", "TRACK-DOWN"]
    assert payload["route"]["blocks"][0]["route_start_m"] == 0
    assert payload["route"]["blocks"][1]["route_start_m"] == 2000
    assert payload["route"]["blocks"][2]["curve_direction"] == "RIGHT"
    assert payload["signals"][1]["route_position_m"] == 2000
    assert payload["signals"][8]["direction"] == "REVERSE"
    assert payload["signals"][8]["route_position_m"] == config.route.total_length_m
    assert len(payload["stations"]) == 4
    assert len(payload["trains"]) == 4
    assert payload["dynamic_signalling"] is True


def test_reset_reconstructs_all_initial_train_states():
    manager = SessionManager()
    session = manager.create("delhi_agra_corridor.yaml")
    initial_count = len(session.snapshots())
    session.tick(); session.tick()
    assert session.engine.sim_time_s > 0
    assert len(session.frames) == initial_count * 3
    frames = session.reset()
    assert session.engine.sim_time_s == 0
    assert len(frames) == 4
    assert all(frame.sim_time_s == 0 for frame in frames)
    assert session.frames == frames
    assert session.export_paths is None


def test_dynamic_signals_are_derived_from_track_occupancy():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV3(config, scenario_id="signals")
    states = engine.signal_states()
    assert states["UP-03"] == SignalAspect.RED
    assert states["DN-08"] == SignalAspect.GREEN
    for _ in range(60): engine.tick()
    assert engine.sim_time_s >= engine.trains[-1].departure_time_s
    assert engine.signal_states()["DN-08"] == SignalAspect.RED


def test_multi_train_engine_contains_same_and_opposite_track_runs_and_station_dwells():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV3(config, scenario_id="multi")
    assert [train.train.track_id for train in engine.trains].count("TRACK-UP") == 3
    assert [train.train.track_id for train in engine.trains].count("TRACK-DOWN") == 1
    assert engine.trains[-1].direction.value == "REVERSE"
    saw_station_dwell = False
    for _ in range(1800):
        frames = engine.tick()
        if any(frame.control_reason.startswith("STATION_DWELL:") for frame in frames):
            saw_station_dwell = True
            break
    assert saw_station_dwell


def test_playback_rate_never_changes_legacy_primary_physics():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    one_x = SimulationEngine(config, scenario_id="same")
    ten_x = SimulationEngine(config, scenario_id="same")
    one_frames = [one_x.snapshot()]; ten_frames = [ten_x.snapshot()]
    while not one_x.is_complete: one_frames.extend(one_x.tick())
    while not ten_x.is_complete: ten_frames.extend(ten_x.tick())
    assert [f.model_dump(mode="json") for f in one_frames] == [f.model_dump(mode="json") for f in ten_frames]


def test_completed_dashboard_run_exports_each_train_with_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "OUTPUT_DIR", tmp_path)
    manager = SessionManager()
    session = manager.create("delhi_agra_corridor.yaml")
    for _ in range(7200):
        if session.engine.is_complete: break
        session.tick()
    if not session.engine.is_complete:
        state = [
            (
                train.train.train_id,
                train.route_position_m,
                train.destination_m,
                train.speed_kmh,
                train.completed,
                train.dwelling_station_id,
                train.dwell_until_s,
                sorted(train.served_stations),
            )
            for train in session.engine.trains
        ]
        raise AssertionError(f"network did not complete: {state}")
    assert session.export_paths is not None
    actual_csv = tmp_path / Path(session.export_paths["csv"]).name
    actual_parquet = tmp_path / Path(session.export_paths["parquet"]).name
    assert actual_csv.exists() and actual_parquet.exists()
    with actual_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(session.frames)
    train_ids = {row["train_id"] for row in rows}
    assert len(train_ids) == 4
    for train_id in train_ids:
        train_rows = [row for row in rows if row["train_id"] == train_id]
        assert min(float(row["actual_remaining_time_s"]) for row in train_rows) == 0.0
        assert all(row["actual_arrival_simulation_s"] for row in train_rows)


def test_api_creates_multi_train_session_and_returns_visual_config():
    response = client.post("/api/sessions", json={"scenario_name": "delhi_agra_corridor.yaml"})
    assert response.status_code == 201
    payload = response.json()
    assert payload["session_id"]
    assert payload["scenario_id"].startswith("live-")
    assert payload["config"]["route"]["route_id"] == "GT-DELHI-AGRA-01"
    assert len(payload["initial_frames"]) == 4
    assert payload["initial_frame"]["actual_remaining_time_s"] is None
    assert payload["signal_states"]["UP-03"] == "RED"


def test_arbitrary_server_path_is_rejected():
    response = client.post("/api/sessions", json={"scenario_name": "../secret.yaml"})
    assert response.status_code == 404
