import sys
sys.path.insert(0, ".")

from simulator.config_loader import load_simulation_config
from simulator.dataset import label_completed_multi_train_journey
from simulator.models import TemporarySpeedRestriction, WeatherCondition
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from simulator.scenario_generator import ScenarioGenerator, ScenarioGeneratorConfig
from simulator.track_aware_exporters import TrackAwareParquetTelemetryExporter
from simulator.track_blocks import enumerate_track_blocks, track_block_id


config = load_simulation_config("examples/delhi_agra_corridor.yaml")

# 1. Canonical corridor shape and safe baseline.
sections = enumerate_track_blocks(config.route)
assert len(config.route.blocks) == 8
assert len(config.route.track_ids) == 2
assert len(sections) == 16
assert [section.track_block_id for section in sections[:8]] == [f"UP-BLK-{i:02d}" for i in range(1, 9)]
assert [section.track_block_id for section in sections[8:]] == [f"DOWN-BLK-{i:02d}" for i in range(1, 9)]
assert config.environment.temporary_speed_restrictions == []
assert config.environment.maintenance_restrictions == []
assert all(
    len(schedule.timeline) == 1
    and schedule.timeline[0].condition == WeatherCondition.CLEAR
    for schedule in config.environment.weather
)

# 2. Full multi-train restrictive-network run.
engine = NetworkSimulationEngineV4Restrictive(config.model_copy(deep=True), scenario_id="smoke-network")
frames = engine.snapshot_all()
while not engine.is_complete and engine.sim_time_s < config.simulation.max_simulation_time_s:
    frames.extend(engine.tick())
assert engine.is_complete
assert len(engine.trains) == 4
assert all(train.completed for train in engine.trains)

# 3. Exact post-run labels are generated independently per train.
labelled = label_completed_multi_train_journey(frames)
train_ids = {frame.train_id for frame in labelled}
assert len(train_ids) == 4
for train_id in train_ids:
    train_frames = [frame for frame in labelled if frame.train_id == train_id]
    assert train_frames
    assert min(frame.actual_remaining_time_s for frame in train_frames) == 0
    assert all(frame.actual_arrival_simulation_s is not None for frame in train_frames)
    assert all(frame.total_journey_time_s is not None for frame in train_frames)

# 4. Tick export preserves both logical block and canonical track-block identity.
telemetry = TrackAwareParquetTelemetryExporter(config).to_dataframe(labelled)
assert "block_id" in telemetry.columns
assert "track_block_id" in telemetry.columns
assert all(
    track_block == track_block_id(track, block)
    for track_block, track, block in zip(
        telemetry["track_block_id"], telemetry["track_id"], telemetry["block_id"]
    )
)

# 5. A track-specific TSR affects only its selected physical track.
restriction_config = config.model_copy(deep=True)
restriction_config.environment.temporary_speed_restrictions = [
    TemporarySpeedRestriction(
        restriction_id="SMOKE-UP-TSR",
        block_id="BLK-02",
        track_id="TRACK-UP",
        start_position_m=600.0,
        end_position_m=900.0,
        speed_limit_kmh=20.0,
        start_time_s=0.0,
        end_time_s=500.0,
    )
]
restriction_engine = NetworkSimulationEngineV4Restrictive(
    restriction_config, scenario_id="smoke-track-restriction"
)
restriction_engine.sim_time_s = 100.0
block_start = restriction_config.route.block_start_distance_m("BLK-02")
up = next(train for train in restriction_engine.trains if train.current_track_id == "TRACK-UP")
down = next(train for train in restriction_engine.trains if train.current_track_id == "TRACK-DOWN")
for train in (up, down):
    train.route_position_m = block_start + 700.0
assert restriction_engine._current_ceiling(up)[0] == 20.0
assert restriction_engine._current_ceiling(down)[0] > 20.0

# 6. Randomized generation uses the same track-aware network architecture.
generator = ScenarioGenerator(
    config,
    ScenarioGeneratorConfig(
        n_scenarios=2,
        random_seed=7,
        baseline_probability=0.0,
        tsr_probability=1.0,
        crossing_closure_probability=0.0,
        signal_restriction_probability=0.0,
    ),
)
exporter = generator.run()
df = exporter.to_dataframe()
assert df["scenario_id"].nunique() == 2
assert df["actual_remaining_time_s"].notna().all()
assert set(df["track_id"]).issubset(set(config.route.track_ids))

print(
    f"network smoke: {len(frames)} raw frames, "
    f"{len(train_ids)} trains, {len(sections)} canonical track-blocks"
)
print(f"generated {len(df)} labelled rows across 2 randomized scenarios")
print("Phase 2 track-block smoke test passed")
