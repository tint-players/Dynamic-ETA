import sys
sys.path.insert(0, ".")

from simulator import (
    load_simulation_config,
    SimulationEngine,
    ScenarioGenerator,
    ScenarioGeneratorConfig,
    label_completed_journey,
)
from simulator.models import SignalAspect, SignalStateSchedule, SignalTimelineEntry

config = load_simulation_config("examples/delhi_agra_corridor.yaml")
assert len(config.route.blocks) == 8
assert len(config.route.track_ids) == 2
assert len(config.signals) == 18

# 1. Full source-to-destination run using the legacy single-train engine.
# This remains a regression check for Component A primary-train physics.
engine = SimulationEngine(config, scenario_id="single_run")
frames = engine.run()
assert engine.is_complete
assert frames[0].sim_time_s == 0
assert frames[-1].distance_to_destination_m == 0
assert frames[-1].speed_kmh == 0
assert all(frames[i].route_position_m <= frames[i + 1].route_position_m for i in range(len(frames) - 1))

# 2. Exact ground-truth remaining-time labels.
labelled = label_completed_journey(frames)
assert labelled[-1].actual_remaining_time_s == 0
assert labelled[0].actual_remaining_time_s == labelled[-1].sim_time_s
assert all(x.actual_remaining_time_s >= y.actual_remaining_time_s for x, y in zip(labelled, labelled[1:]))

# 3. TSR pre-braking.
tsr = config.environment.temporary_speed_restrictions[0]
tsr_start = config.route.block_start_distance_m(tsr.block_id) + tsr.start_position_m
near_tsr = min(frames, key=lambda f: abs(f.route_position_m - tsr_start))
assert near_tsr.speed_kmh <= tsr.speed_limit_kmh + 2.0

# 4. RED signal stop/release behavior on the forward/up track.
red_config = load_simulation_config("examples/delhi_agra_corridor.yaml")
for i, schedule in enumerate(red_config.environment.signal_states):
    if schedule.signal_id == "UP-02":
        red_config.environment.signal_states[i] = SignalStateSchedule(
            signal_id="UP-02",
            timeline=[
                SignalTimelineEntry(start_time_s=0, aspect=SignalAspect.RED),
                SignalTimelineEntry(start_time_s=180, aspect=SignalAspect.GREEN),
            ],
        )
red_engine = SimulationEngine(red_config, scenario_id="red_signal_test")
red_frames = []
for _ in range(180):
    red_frames.extend(red_engine.tick())
boundary = red_config.route.block_start_distance_m("BLK-02")
assert max(f.route_position_m for f in red_frames) < boundary
assert any(f.control_reason.startswith("RED_SIGNAL") and f.speed_kmh == 0 for f in red_frames)
for _ in range(100):
    red_frames.extend(red_engine.tick())
assert any(f.route_position_m > boundary for f in red_frames if f.sim_time_s > 180)

# 5. Generate multiple normal/controlled scenarios with labels.
generator = ScenarioGenerator(config, ScenarioGeneratorConfig(n_scenarios=5, random_seed=7))
exporter = generator.run()
df = exporter.to_dataframe()
assert df["scenario_id"].nunique() == 5
assert df["actual_remaining_time_s"].notna().all()
assert (df.groupby("scenario_id").tail(1)["actual_remaining_time_s"] == 0).all()

exporter.to_csv("output/sample_training_data.csv")
exporter.to_parquet("output/sample_training_data.parquet")

print(f"single run: {len(frames)} frames, arrival={frames[-1].sim_time_s:.1f}s")
print(f"generated {len(df)} labelled rows across 5 scenarios")
print(df[["scenario_id", "sim_time_s", "current_block_id", "speed_kmh", "weather", "actual_remaining_time_s"]].head())
print("Component A smoke test passed")
