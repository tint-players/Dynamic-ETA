from __future__ import annotations

from backend.session import SessionManager
from simulator.models import SignalAspect
from simulator.network_engine_v4 import NetworkSimulationEngineV4
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def test_agra_dn_signals_follow_forward_crossover_exception_then_return_to_dynamic():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="agra-v7-signals")

    cross_up = next(train for train in engine.trains if train.train.train_id == "TRAIN-CROSS-UP")
    turnaround = next(train for train in engine.trains if train.train.train_id == "TRAIN-12002")
    for train in engine.trains:
        if train not in (cross_up, turnaround):
            train.completed = True
    turnaround.completed = True

    dn07 = next(signal for signal in config.signals if signal.signal_id == "DN-07")
    dn08 = next(signal for signal in config.signals if signal.signal_id == "DN-08")
    crossover = next(item for item in config.crossovers if item.crossover_id == "XOVER-AGRA-01")
    _, crossover_end, midpoint = engine._crossover_bounds(crossover)

    # Before TRAIN-CROSS-UP has actually taken the crossover, both signals retain
    # the existing dynamic-signalling result.
    cross_up.current_track_id = crossover.from_track_id
    cross_up.completed_crossovers.clear()
    cross_up.source_m = midpoint - 1000
    cross_up.destination_m = config.route.total_length_m
    cross_up.route_position_m = midpoint - 50
    assert engine.signal_aspect(dn07) == NetworkSimulationEngineV4.signal_aspect(engine, dn07)
    assert engine.signal_aspect(dn08) == NetworkSimulationEngineV4.signal_aspect(engine, dn08)

    # Once the non-reversing UP train has crossed onto TRACK-DOWN and continues
    # right, DN-08 is held RED while DN-07 gives the requested YELLOW indication.
    reason, turned_around = engine._apply_track_change_v4(cross_up, midpoint - 1, midpoint + 1)
    assert reason == "CROSSOVER:XOVER-AGRA-01"
    assert turned_around is False
    cross_up.route_position_m = crossover_end + 100
    assert engine.signal_aspect(dn07) == SignalAspect.YELLOW
    assert engine.signal_aspect(dn08) == SignalAspect.RED

    # If TRAIN-12002 also occupies BLK-07 on TRACK-DOWN, DN-07 must remain RED
    # for occupied-block protection; DN-08 stays RED under the crossover exception.
    turnaround.completed = False
    turnaround.departure_time_s = 0
    turnaround.current_track_id = "TRACK-DOWN"
    block07_start = config.route.block_start_distance_m("BLK-07")
    turnaround.route_position_m = block07_start + 2800
    turnaround.source_m = turnaround.route_position_m + 500
    turnaround.destination_m = 0
    assert engine.signal_aspect(dn07) == SignalAspect.RED
    assert engine.signal_aspect(dn08) == SignalAspect.RED

    # The DN-08 rule is conditional, not permanent. Once TRAIN-CROSS-UP is no
    # longer active, the normal dynamic result is used again.
    cross_up.completed = True
    assert engine.signal_aspect(dn08) == NetworkSimulationEngineV4.signal_aspect(engine, dn08)
