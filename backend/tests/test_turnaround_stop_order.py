from backend.session import SessionManager
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def _station_reasons(engine, train):
    return [target.reason for target in engine._targets(train) if target.reason.startswith("STATION:")]


def test_turnaround_train_defers_ballabgarh_until_return_leg():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="return-stop-order")
    train = next(item for item in engine.trains if item.train.train_id == "TRAIN-12002")

    assert [stop.station_id for stop in train.station_stops] == [
        "MATHURA",
        "FARAH",
        "AGRA-CANTT",
        "BALLABGARH",
    ]
    assert _station_reasons(engine, train) == ["STATION:MATHURA"]

    # Emulate the state immediately after the Agra crossover turnaround. The
    # already-served outbound stations stay served; Ballabgarh is now the next
    # configured stop and lies ahead on TRACK-DOWN in the reverse direction.
    train.current_track_id = "TRACK-DOWN"
    train.route_position_m = 15_400.0
    train.source_m = 15_400.0
    train.destination_m = 0.0
    train.served_stations.update({"MATHURA", "FARAH", "AGRA-CANTT"})
    train.completed_crossovers.add("XOVER-AGRA-01")

    assert _station_reasons(engine, train) == ["STATION:BALLABGARH"]
