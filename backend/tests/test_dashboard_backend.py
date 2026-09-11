from fastapi.testclient import TestClient

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

    frame = session.reset()
    assert session.engine.sim_time_s == 0
    assert frame.sim_time_s == 0
    assert frame.route_position_m == 0
    assert frame.speed_kmh == 0


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
