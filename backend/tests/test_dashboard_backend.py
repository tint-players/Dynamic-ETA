import csv
from pathlib import Path

from fastapi.testclient import TestClient

import backend.session as session_module
from backend.main import app
from backend.session import SessionManager
from backend.viz import config_for_visualization
from simulator.engine import SimulationEngine


client = TestClient(app)


def test_visual_config_uses_backend_route_geometry():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    payload = config_for_visualization(config)

    assert payload["route"]["blocks"][0]["route_start_m"] == 0
    assert payload["route"]["blocks"][1]["route_start_m"] == 2000
    assert payload["signals"][1]["route_position_m"] == 2000
    assert payload["environment"]["temporary_speed_restrictions"][0]["route_start_m"] == 5000
    assert payload["environment"]["temporary_speed_restrictions"][0]["route_end_m"] == 5700


def test_reset_reconstructs_initial_engine_state():
    manager = SessionManager()
    session = manager.create("delhi_agra_corridor.yaml")
    session.tick()
    session.tick()
    assert session.engine.sim_time_s > 0
    assert len(session.frames) == 3

    frame = session.reset()
    assert session.engine.sim_time_s == 0
    assert frame.sim_time_s == 0
    assert frame.route_position_m == 0
    assert frame.speed_kmh == 0
    assert session.frames == [frame]
    assert session.export_paths is None


def test_playback_rate_never_changes_simulation_telemetry():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    one_x = SimulationEngine(config, scenario_id="same")
    ten_x = SimulationEngine(config, scenario_id="same")

    one_frames = [one_x.snapshot()]
    ten_frames = [ten_x.snapshot()]
    while not one_x.is_complete:
        one_frames.extend(one_x.tick())
    while not ten_x.is_complete:
        ten_frames.extend(ten_x.tick())

    assert [f.model_dump(mode="json") for f in one_frames] == [
        f.model_dump(mode="json") for f in ten_frames
    ]


def test_completed_dashboard_run_exports_labelled_csv_and_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "OUTPUT_DIR", tmp_path)
    manager = SessionManager()
    session = manager.create("delhi_agra_corridor.yaml")

    while not session.engine.is_complete:
        session.tick()

    assert session.export_paths is not None
    csv_path = Path(__file__).resolve().parents[2] / session.export_paths["csv"]
    parquet_path = Path(__file__).resolve().parents[2] / session.export_paths["parquet"]

    # export_paths are repo-relative in production; with a monkeypatched temp output,
    # validate the actual temp files by filename.
    actual_csv = tmp_path / Path(session.export_paths["csv"]).name
    actual_parquet = tmp_path / Path(session.export_paths["parquet"]).name
    assert actual_csv.exists()
    assert actual_parquet.exists()

    with actual_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(session.frames)
    assert float(rows[0]["actual_remaining_time_s"]) == session.frames[-1].sim_time_s
    assert float(rows[-1]["actual_remaining_time_s"]) == 0.0
    assert float(rows[-1]["actual_arrival_simulation_s"]) == session.frames[-1].sim_time_s
    assert csv_path.name == actual_csv.name
    assert parquet_path.name == actual_parquet.name


def test_api_creates_session_and_returns_visual_config():
    response = client.post(
        "/api/sessions", json={"scenario_name": "delhi_agra_corridor.yaml"}
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["session_id"]
    assert payload["scenario_id"].startswith("live-")
    assert payload["config"]["route"]["route_id"] == "GT-DELHI-AGRA-01"
    assert payload["initial_frame"]["actual_remaining_time_s"] is None


def test_arbitrary_server_path_is_rejected():
    response = client.post("/api/sessions", json={"scenario_name": "../secret.yaml"})
    assert response.status_code == 404
