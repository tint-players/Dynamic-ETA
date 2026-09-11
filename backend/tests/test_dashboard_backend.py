import csv
from pathlib import Path

from fastapi.testclient import TestClient

import backend.session as session_module
from backend.main import app
from backend.session import SessionManager
from backend.viz import config_for_visualization
from simulator.engine import SimulationEngine
from simulator.models import SignalAspect
from simulator.network_engine import NetworkSimulationEngine


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
    assert len(payload["crossovers"]) == 1
    assert payload["crossovers"][0]["from_track_id"] == "TRACK-UP"
    assert payload["crossovers"][0]["to_track_id"] == "TRACK-DOWN"
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
    engine = NetworkSimulationEngine(config, scenario_id="signals")
    states = engine.signal_states()
    assert states["UP-03"] == SignalAspect.RED
    assert states["DN-08"] == SignalAspect.GREEN
    for _ in range(60): engine.tick()
    down_train = next(t for t in engine.trains if t.train.train_id == "TRAIN-DOWN-01")
    assert engine.sim_time_s >= down_train.departure_time_s
    assert engine.signal_states()["DN-08"] == SignalAspect.RED


def test_fleet_is_three_up_one_down_and_middle_up_train_changes_track_at_agra():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngine(config, scenario_id="multi")
    assert sum(t.direction.value == "FORWARD" for t in engine.trains) == 3
    assert sum(t.direction.value == "REVERSE" for t in engine.trains) == 1
    assert [t.train.track_id for t in engine.trains].count("TRACK-UP") == 3
    assert [t.train.track_id for t in engine.trains].count("TRACK-DOWN") == 1

    crossover_train = next(t for t in engine.trains if t.train.train_id == "TRAIN-CROSS-UP")
    assert crossover_train.current_track_id == "TRACK-UP"
    saw_station_dwell = False
    changed_track = False
    for _ in range(2400):
        frames = engine.tick()
        if any(frame.control_reason.startswith("STATION_DWELL:") for frame in frames):
            saw_station_dwell = True
        frame = next(f for f in frames if f.train_id == "TRAIN-CROSS-UP")
        if frame.track_id == "TRACK-DOWN":
            changed_track = True
            break
    assert saw_station_dwell
    assert changed_track
    assert "XOVER-AGRA-01" in crossover_train.completed_crossovers


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
    assert session.engine.is_complete
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
