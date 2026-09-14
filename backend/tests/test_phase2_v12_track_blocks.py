from __future__ import annotations

import pytest

from backend.session import SessionManager
from simulator.models import TemporarySpeedRestriction
from simulator.track_blocks import enumerate_track_blocks, get_track_block, locate_track_block, track_block_id
from simulator.track_block_validation import validate_track_block_config


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _route():
    return _config().route


def test_delhi_agra_expands_to_sixteen_track_blocks_in_stable_order():
    sections = enumerate_track_blocks(_route())

    assert len(sections) == 16
    assert [section.track_block_id for section in sections[:8]] == [f"UP-BLK-{i:02d}" for i in range(1, 9)]
    assert [section.track_block_id for section in sections[8:]] == [f"DOWN-BLK-{i:02d}" for i in range(1, 9)]
    assert len({section.track_block_id for section in sections}) == 16


def test_up_and_down_sections_share_geometry_but_have_distinct_operational_identity():
    route = _route()
    up = get_track_block(route, "TRACK-UP", "BLK-05")
    down = get_track_block(route, "TRACK-DOWN", "BLK-05")

    assert up.track_block_id == "UP-BLK-05"
    assert down.track_block_id == "DOWN-BLK-05"
    assert up.geometry is down.geometry
    assert up.route_start_m == down.route_start_m
    assert up.route_end_m == down.route_end_m
    assert up.track_id != down.track_id


def test_locate_track_block_preserves_shared_route_geometry_and_selected_track():
    route = _route()
    start = route.block_start_distance_m("BLK-03")

    section, local = locate_track_block(route, "TRACK-DOWN", start + 350.0)

    assert section.track_block_id == "DOWN-BLK-03"
    assert section.block_id == "BLK-03"
    assert local == pytest.approx(350.0)


def test_unknown_track_or_block_is_rejected():
    route = _route()

    with pytest.raises(KeyError, match="Unknown track_id"):
        get_track_block(route, "TRACK-NOT-REAL", "BLK-01")
    with pytest.raises(KeyError, match="Unknown block_id"):
        get_track_block(route, "TRACK-UP", "BLK-99")


def test_track_block_id_is_stable_and_validates_empty_components():
    assert track_block_id("TRACK-UP", "BLK-07") == "UP-BLK-07"
    assert track_block_id("TRACK-DOWN", "BLK-07") == "DOWN-BLK-07"

    with pytest.raises(ValueError, match="track_id"):
        track_block_id("", "BLK-01")
    with pytest.raises(ValueError, match="block_id"):
        track_block_id("TRACK-UP", "")


def test_multitrack_restriction_requires_explicit_track():
    config = _config()
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="AMBIGUOUS",
            block_id="BLK-02",
            start_position_m=100.0,
            end_position_m=200.0,
            speed_limit_kmh=40.0,
        )
    ]

    with pytest.raises(ValueError, match="must specify track_id"):
        validate_track_block_config(config)


def test_track_block_validator_accepts_track_specific_restriction_and_shared_weather():
    config = _config()
    config.environment.temporary_speed_restrictions = [
        TemporarySpeedRestriction(
            restriction_id="UP-ONLY",
            block_id="BLK-02",
            track_id="TRACK-UP",
            start_position_m=100.0,
            end_position_m=200.0,
            speed_limit_kmh=40.0,
        )
    ]

    assert validate_track_block_config(config) is config
    assert config.environment.weather[1].block_id == "BLK-02"
