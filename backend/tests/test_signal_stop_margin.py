from backend.session import SessionManager
from simulator.models import SignalAspect
from simulator.network_engine_v4 import NetworkSimulationEngineV4


def test_up06_red_signal_stops_before_post():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV4(config, scenario_id="signal-stop-margin")
    signal = next(item for item in config.signals if item.signal_id == "UP-06")
    signal_position = engine._signal_position(signal)

    train = next(t for t in engine.trains if t.train.train_id == "TRAIN-12002")
    blocker = next(t for t in engine.trains if t.train.train_id == "TRAIN-CROSS-UP")

    for other in engine.trains:
        if other is not train and other is not blocker:
            other.completed = True

    train.departure_time_s = 0
    train.current_track_id = "TRACK-UP"
    train.route_position_m = signal_position - 450.0
    train.source_m = train.route_position_m
    train.destination_m = config.route.total_length_m
    train.speed_kmh = 90.0
    train.station_stops = []

    blocker.departure_time_s = 0
    blocker.current_track_id = "TRACK-UP"
    blocker.route_position_m = signal_position + 700.0
    blocker.source_m = blocker.route_position_m
    blocker.destination_m = config.route.total_length_m
    blocker.speed_kmh = 0.0
    blocker.station_stops = []
    blocker.track_changes = []

    assert engine.signal_aspect(signal, exclude=train) == SignalAspect.RED
    targets = [target for target in engine._targets(train) if target.reason == "RED_SIGNAL:UP-06"]
    assert len(targets) == 1
    expected_stop = signal_position - engine.SIGNAL_STOP_MARGIN_M
    assert targets[0].position_m == expected_stop

    for _ in range(180):
        engine.tick()
        assert train.route_position_m <= expected_stop + 1e-6
        if abs(train.route_position_m - expected_stop) <= 0.02:
            assert train.speed_kmh == 0.0
            break
    else:
        raise AssertionError("train did not reach the protected stop line before UP-06")


def test_reverse_red_signal_stop_line_is_before_post_in_reverse_direction():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV4(config, scenario_id="reverse-signal-stop-margin")
    signal = next(item for item in config.signals if item.signal_id == "DN-06")
    signal_position = engine._signal_position(signal)

    train = next(t for t in engine.trains if t.train.train_id == "TRAIN-DOWN-01")
    for other in engine.trains:
        if other is not train:
            other.completed = True

    train.departure_time_s = 0
    train.current_track_id = "TRACK-DOWN"
    train.route_position_m = signal_position + 350.0
    train.source_m = train.route_position_m
    train.destination_m = 0.0
    train.speed_kmh = 70.0
    train.station_stops = []

    # Force the signal to RED through occupancy in its protected block using a
    # temporary blocker with the same physical track.
    blocker = next(t for t in engine.trains if t.train.train_id != train.train.train_id)
    blocker.completed = False
    blocker.departure_time_s = 0
    blocker.current_track_id = "TRACK-DOWN"
    blocker.route_position_m = signal_position - 300.0
    blocker.source_m = blocker.route_position_m
    blocker.destination_m = 0.0
    blocker.speed_kmh = 0.0
    blocker.station_stops = []
    blocker.track_changes = []

    assert engine.signal_aspect(signal, exclude=train) == SignalAspect.RED
    targets = [target for target in engine._targets(train) if target.reason == "RED_SIGNAL:DN-06"]
    assert len(targets) == 1
    expected_stop = signal_position + engine.SIGNAL_STOP_MARGIN_M
    assert targets[0].position_m == expected_stop
