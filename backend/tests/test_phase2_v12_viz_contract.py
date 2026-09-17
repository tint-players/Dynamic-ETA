from __future__ import annotations

from backend.session import SessionManager
from backend.viz import config_for_visualization


def _viz():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml")
    return config_for_visualization(config)


def test_visualization_exposes_sixteen_canonical_track_blocks():
    viz = _viz()
    track_blocks = viz["route"]["track_blocks"]

    assert len(track_blocks) == 16
    assert [item["track_block_id"] for item in track_blocks[:8]] == [f"UP-BLK-{i:02d}" for i in range(1, 9)]
    assert [item["track_block_id"] for item in track_blocks[8:]] == [f"DOWN-BLK-{i:02d}" for i in range(1, 9)]


def test_signals_platforms_and_crossover_reference_canonical_track_blocks():
    viz = _viz()

    up05 = next(item for item in viz["signals"] if item["signal_id"] == "UP-05")
    assert up05["protected_track_block_id"] == "UP-BLK-05"

    farah = next(item for item in viz["stations"] if item["station_id"] == "FARAH")
    assert {platform["track_block_id"] for platform in farah["platforms"]} == {
        "UP-BLK-05",
        "DOWN-BLK-05",
    }

    crossover = viz["crossovers"][0]
    assert crossover["from_track_block_id"] == "UP-BLK-07"
    assert crossover["to_track_block_id"] == "DOWN-BLK-07"


def test_crossing_is_explicit_multi_track_infrastructure():
    viz = _viz()
    crossing = next(item for item in viz["environment"]["crossings"] if item["crossing_id"] == "XING-001")

    assert crossing["block_id"] == "BLK-05"
    assert crossing["track_ids"] == ["TRACK-UP", "TRACK-DOWN"]
    assert crossing["affected_track_block_ids"] == ["UP-BLK-05", "DOWN-BLK-05"]


def test_weather_remains_shared_logical_block_state():
    viz = _viz()
    weather = next(item for item in viz["environment"]["weather"] if item["block_id"] == "BLK-05")

    assert "track_id" not in weather
    assert "track_block_id" not in weather
