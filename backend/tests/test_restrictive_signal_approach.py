from backend.session import SessionManager
from simulator.models import SignalAspect
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def _isolated_up_train_and_blocker():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="restrictive-signal")
    train = next(t for t in engine.trains if t.train.train_id == "TRAIN-12002")
    blocker = next(t for t in engine.trains if t.train.train_id == "TRAIN-CROSS-UP")
    for other in engine.trains:
        if other is not train and other is not blocker:
            other.completed = True

    train.departure_time_s = 0
    train.current_track_id = "TRACK-UP"
    train.station_stops = []
    train.track_changes = []

    blocker.departure_time_s = 0
    blocker.current_track_id = "TRACK-UP"
    blocker.station_stops = []
    blocker.track_changes = []
    blocker.speed_kmh = 0.0
    return config, engine, train, blocker


def test_yellow_signal_caps_approach_speed_instead_of_reaccelerating_to_line_speed():
    config, engine, train, blocker = _isolated_up_train_and_blocker()
    signal = next(item for item in config.signals if item.signal_id == "UP-07")
    signal_position = engine._signal_position(signal)
    next_block_start = config.route.block_start_distance_m("BLK-08")

    train.route_position_m = signal_position - 500.0
    train.source_m = train.route_position_m
    train.destination_m = config.route.total_length_m
    train.speed_kmh = 60.0

    blocker.route_position_m = next_block_start + 500.0
    blocker.source_m = blocker.route_position_m
    blocker.destination_m = config.route.total_length_m

    assert engine.signal_aspect(signal, exclude=train) == SignalAspect.YELLOW
    ceiling, weather, _ = engine._current_ceiling(train)
    desired, reason = engine._desired_speed(train, ceiling, weather)
    assert desired == engine.YELLOW_APPROACH_SPEED_KMH
    assert reason == "YELLOW_APPROACH:UP-07"

    engine.tick()
    assert train.speed_kmh <= engine.YELLOW_APPROACH_SPEED_KMH + 1e-6


def test_yellow_approach_speed_holds_after_passing_yellow_until_next_restrictive_signal():
    config, engine, train, blocker = _isolated_up_train_and_blocker()
    yellow = next(item for item in config.signals if item.signal_id == "UP-07")
    red = next(item for item in config.signals if item.signal_id == "UP-08")
    yellow_position = engine._signal_position(yellow)
    red_position = engine._signal_position(red)

    train.route_position_m = yellow_position + 100.0
    train.source_m = train.route_position_m
    train.destination_m = config.route.total_length_m
    train.speed_kmh = 60.0

    blocker.route_position_m = red_position + 400.0
    blocker.source_m = blocker.route_position_m
    blocker.destination_m = config.route.total_length_m

    assert engine.signal_aspect(yellow, exclude=train) == SignalAspect.YELLOW
    assert engine.signal_aspect(red, exclude=train) == SignalAspect.RED
    ceiling, weather, _ = engine._current_ceiling(train)
    desired, reason = engine._desired_speed(train, ceiling, weather)
    assert desired <= engine.YELLOW_APPROACH_SPEED_KMH
    assert reason in {"YELLOW_APPROACH:UP-07", "RED_SIGNAL:UP-08"}


def test_restrictive_approach_cap_releases_when_route_ahead_is_green():
    config, engine, train, blocker = _isolated_up_train_and_blocker()
    signal = next(item for item in config.signals if item.signal_id == "UP-07")
    signal_position = engine._signal_position(signal)

    train.route_position_m = signal_position - 500.0
    train.source_m = train.route_position_m
    train.destination_m = config.route.total_length_m
    train.speed_kmh = 60.0
    blocker.completed = True

    assert engine.signal_aspect(signal, exclude=train) == SignalAspect.GREEN
    ceiling, weather, _ = engine._current_ceiling(train)
    desired, reason = engine._desired_speed(train, ceiling, weather)
    assert desired > engine.YELLOW_APPROACH_SPEED_KMH
    assert not reason.startswith("YELLOW_APPROACH:")
