import csv
from pathlib import Path

from fastapi.testclient import TestClient

import backend.session as session_module
from backend.main import app
from backend.session import SessionManager
from backend.viz import config_for_visualization
from simulator.engine import SimulationEngine
from simulator.models import CrossingState, SignalAspect
from simulator.network_engine_v4 import NetworkSimulationEngineV4


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
    assert payload["trains"][0]["track_changes"][0]["reverse_after_change"] is True
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
    engine = NetworkSimulationEngineV4(config, scenario_id="signals")
    states = engine.signal_states()
    assert states["UP-03"] == SignalAspect.RED
    assert states["DN-08"] == SignalAspect.GREEN
    for _ in range(60):
        engine.tick()
    down_train = next(t for t in engine.trains if t.train.train_id == "TRAIN-DOWN-01")
    assert engine.sim_time_s >= down_train.departure_time_s
    assert engine.signal_states()["DN-08"] == SignalAspect.RED


def test_level_crossing_closes_for_approach_and_stop_target_is_before_road():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV4(config, scenario_id="crossing")
    crossing = config.environment.crossings[0]
    crossing_pos = config.route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m
    train = engine.trains[0]
    for other in engine.trains[1:]:
        other.completed = True
    train.route_position_m = crossing_pos - 500
    train.source_m = train.route_position_m
    train.destination_m = config.route.total_length_m
    train.current_track_id = "TRACK-UP"
    engine._update_crossings()
    assert engine.crossing_phases()[crossing.crossing_id] == "CLOSING"
    assert engine.crossing_states()[crossing.crossing_id] == CrossingState.CLOSED_FOR_TRAIN
    crossing_targets = [target for target in engine._targets(train) if target.reason == f"CROSSING:{crossing.crossing_id}"]
    assert len(crossing_targets) == 1
    assert crossing_targets[0].position_m == crossing_pos - engine.CROSSING_STOP_LINE_M

    engine.sim_time_s += engine.CROSSING_GATE_CLOSING_S
    engine._update_crossings()
    assert engine.crossing_states()[crossing.crossing_id] == CrossingState.OPEN_FOR_TRAIN
    assert engine.crossing_phases()[crossing.crossing_id] == "PROTECTED"


def test_level_crossing_stays_road_closed_until_rear_clearance_and_delay():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV4(config, scenario_id="clearance")
    crossing = config.environment.crossings[0]
    crossing_pos = config.route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m
    train = engine.trains[0]
    for other in engine.trains[1:]:
        other.completed = True
    train.current_track_id = "TRACK-UP"
    train.route_position_m = crossing_pos - 100
    train.source_m = train.route_position_m
    train.destination_m = config.route.total_length_m
    engine._update_crossings()
    engine.sim_time_s += engine.CROSSING_GATE_CLOSING_S
    engine._update_crossings()
    assert engine.crossing_states()[crossing.crossing_id] == CrossingState.OPEN_FOR_TRAIN

    train.route_position_m = crossing_pos + train.train.length_m + engine.CROSSING_CLEARANCE_MARGIN_M + 1
    engine._update_crossings()
    assert engine.crossing_phases()[crossing.crossing_id] == "CLEARING"
    assert engine.crossing_states()[crossing.crossing_id] == CrossingState.OPEN_FOR_TRAIN
    engine.sim_time_s += engine.CROSSING_REOPEN_DELAY_S - 0.1
    engine._update_crossings()
    assert engine.crossing_states()[crossing.crossing_id] == CrossingState.OPEN_FOR_TRAIN
    engine.sim_time_s += 0.1
    engine._update_crossings()
    assert engine.crossing_states()[crossing.crossing_id] == CrossingState.CLOSED_FOR_TRAIN
    assert engine.crossing_phases()[crossing.crossing_id] == "ROAD_OPEN"


def test_three_up_one_down_first_crossover_and_middle_turnaround():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV4(config, scenario_id="multi")
    assert sum(t.direction.value == "FORWARD" for t in engine.trains) == 3
    assert sum(t.direction.value == "REVERSE" for t in engine.trains) == 1
    assert sum(t.current_track_id == "TRACK-UP" for t in engine.trains) == 3
    assert sum(t.current_track_id == "TRACK-DOWN" for t in engine.trains) == 1

    lead = next(t for t in engine.trains if t.train.train_id == "TRAIN-CROSS-UP")
    middle = next(t for t in engine.trains if t.train.train_id == "TRAIN-12002")
    follower = next(t for t in engine.trains if t.train.train_id == "TRAIN-FOLLOW-UP")

    lead_crossed = False
    middle_turned = False
    for _ in range(7200):
        if engine.is_complete:
            break
        frames = engine.tick()
        lead_frame = next(f for f in frames if f.train_id == "TRAIN-CROSS-UP")
        middle_frame = next(f for f in frames if f.train_id == "TRAIN-12002")
        if lead_frame.track_id == "TRACK-DOWN":
            lead_crossed = True
            assert lead_frame.direction.value == "FORWARD"
        if middle_frame.track_id == "TRACK-DOWN" and middle_frame.direction.value == "REVERSE":
            middle_turned = True

    assert lead_crossed
    assert middle_turned
    assert "XOVER-AGRA-01" in lead.completed_crossovers
    assert "XOVER-AGRA-01" in middle.completed_crossovers
    assert engine.is_complete
    assert all(t.completed for t in engine.trains)
    assert follower.route_position_m == config.route.total_length_m
    assert middle.route_position_m == 0


def test_playback_rate_never_changes_legacy_primary_physics():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    one_x = SimulationEngine(config, scenario_id="same")
    ten_x = SimulationEngine(config, scenario_id="same")
    one_frames = [one_x.snapshot()]; ten_frames = [ten_x.snapshot()]
    while not one_x.is_complete: one_frames.extend(one_x.tick())
    while not ten_x.is_complete: ten_frames.extend(ten_x.tick())
    assert [f.model_dump(mode="json") for f in one_frames] == [f.model_dump(mode="json") for f in ten_frames]


def test_completed_dashboard_run_exports_only_after_all_trains_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "OUTPUT_DIR", tmp_path)
    manager = SessionManager()
    session = manager.create("delhi_agra_corridor.yaml")
    saw_partial_completion = False
    for _ in range(7200):
        if session.engine.is_complete:
            break
        session.tick()
        completed_count = sum(t.completed for t in session.engine.trains)
        if 0 < completed_count < len(session.engine.trains):
            saw_partial_completion = True
            assert session.export_paths is None
    assert saw_partial_completion
    assert session.engine.is_complete
    assert all(t.completed for t in session.engine.trains)
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
    assert set(payload["crossing_states"]) == {"XING-001", "XING-002"}
    assert all(state == "CLOSED_FOR_TRAIN" for state in payload["crossing_states"].values())


def test_arbitrary_server_path_is_rejected():
    response = client.post("/api/sessions", json={"scenario_name": "../secret.yaml"})
    assert response.status_code == 404
