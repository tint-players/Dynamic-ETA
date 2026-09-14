from __future__ import annotations

import pytest

from backend.session import SessionManager
from simulator.models import TemporarySpeedRestriction
from simulator.multi_engine import MultiTrainSimulationEngine


def _engine():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="TSR-UP-ONLY",
            block_id="BLK-02",
            track_id="TRACK-UP",
            start_position_m=600.0,
            end_position_m=900.0,
            speed_limit_kmh=20.0,
            start_time_s=0.0,
            end_time_s=500.0,
        )
    ]
    engine = MultiTrainSimulationEngine(config, scenario_id="legacy-track-restrictions")
    engine.sim_time_s = 100.0
    return config, engine


def test_legacy_multi_engine_current_ceiling_respects_restriction_track():
    config, engine = _engine()
    block_start = config.route.block_start_distance_m("BLK-02")
    up = next(train for train in engine.trains if train.train.track_id == "TRACK-UP")
    down = next(train for train in engine.trains if train.train.track_id == "TRACK-DOWN")

    for train in (up, down):
        train.route_position_m = block_start + 700.0

    assert engine._current_ceiling(up)[0] == pytest.approx(20.0)
    assert engine._current_ceiling(down)[0] > 20.0


def test_legacy_multi_engine_tsr_lookahead_respects_restriction_track():
    config, engine = _engine()
    block_start = config.route.block_start_distance_m("BLK-02")
    up = next(train for train in engine.trains if train.train.track_id == "TRACK-UP")
    down = next(train for train in engine.trains if train.train.track_id == "TRACK-DOWN")

    for train in (up, down):
        train.route_position_m = block_start + 100.0

    up_frame = engine.snapshot_train(up)
    down_frame = engine.snapshot_train(down)

    assert up_frame.next_tsr_limit_kmh == pytest.approx(20.0)
    assert up_frame.distance_to_next_tsr_m == pytest.approx(500.0)
    assert down_frame.next_tsr_limit_kmh is None
    assert down_frame.distance_to_next_tsr_m is None
