from __future__ import annotations

from backend.session import SessionManager
from simulator.models import SignalAspect
from simulator.network_engine_v4 import NetworkSimulationEngineV4
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def test_dn07_returns_to_dynamic_after_train_12002_reverses_while_dn08_stays_red_for_cross_up():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="agra-v9-dn07-dynamic")

    cross_up = next(train for train in engine.trains if train.train.train_id == "TRAIN-CROSS-UP")
    turnaround = next(train for train in engine.trains if train.train.train_id == "TRAIN-12002")
    for train in engine.trains:
        if train not in (cross_up, turnaround):
            train.completed = True

    dn07 = next(signal for signal in config.signals if signal.signal_id == "DN-07")
    dn08 = next(signal for signal in config.signals if signal.signal_id == "DN-08")
    crossover = next(item for item in config.crossovers if item.crossover_id == "XOVER-AGRA-01")
    _, crossover_end, midpoint = engine._crossover_bounds(crossover)

    cross_up.current_track_id = crossover.from_track_id
    cross_up.completed_crossovers.clear()
    cross_up.source_m = midpoint - 1000
    cross_up.destination_m = config.route.total_length_m
    cross_up.route_position_m = midpoint - 1
    engine._apply_track_change_v4(cross_up, midpoint - 1, midpoint + 1)
    cross_up.route_position_m = crossover_end + 100

    turnaround.completed = True
    assert engine.signal_aspect(dn07) == SignalAspect.YELLOW
    assert engine.signal_aspect(dn08) == SignalAspect.RED

    turnaround.completed = False
    turnaround.departure_time_s = 0
    turnaround.current_track_id = "TRACK-DOWN"
    turnaround.completed_crossovers.add("XOVER-AGRA-01")
    turnaround.pending_turnaround_crossover_id = None
    turnaround.source_m = config.route.total_length_m - 500
    turnaround.destination_m = 0
    turnaround.route_position_m = config.route.block_start_distance_m("BLK-06") + 100

    assert turnaround.sign < 0
    assert engine.signal_aspect(dn07) == NetworkSimulationEngineV4.signal_aspect(engine, dn07)
    assert engine.signal_aspect(dn08) == SignalAspect.RED
