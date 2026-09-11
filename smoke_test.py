import sys
sys.path.insert(0, ".")

from simulator import (
    load_simulation_config,
    SimulationEngine,
    ScenarioGenerator,
    ScenarioGeneratorConfig,
    LiveExporter,
    AnomalyInjector,
    AnomalyRequest,
    AnomalyType,
    SignalAspect,
)

# 1. Load config from YAML
config = load_simulation_config("examples/delhi_agra_corridor.yaml")
print(f"Loaded corridor '{config.corridor.name}' with {len(config.corridor.blocks)} blocks, "
      f"{len(config.trains)} train(s)")

# 2. Live-mode style: run engine tick by tick, JSON-serialize a frame
engine = SimulationEngine(config, scenario_id="live_demo")
for _ in range(5):
    frames = engine.tick()
print("Sample live JSON frame:")
print(LiveExporter.to_json(frames[0]))

# 3. Guardrail test: try to force a RED signal on the immediate next block
#    at high speed -> should be rejected and redirected
injector = AnomalyInjector(config.corridor, config.caution_speed_kmh)
train_id = config.trains[0].train_id
state = engine.train_states[train_id]
current_block = config.corridor.blocks[state.block_index]
print(f"\nTrain at block {current_block.block_id}, speed={state.speed_kmh:.1f} km/h")

result = injector.inject(
    AnomalyRequest(
        anomaly_type=AnomalyType.SIGNAL_ASPECT_CHANGE,
        block_id=config.corridor.blocks[state.block_index + 1].block_id,
        signal_aspect=SignalAspect.RED,
    ),
    train_speed_kmh=state.speed_kmh,
    train_current_block_id=current_block.block_id,
    emergency_decel_ms2=state.config.emergency_decel_ms2,
)
print(f"Guardrail result: allowed={result.guardrail.allowed}, event={result.guardrail.event}, "
      f"redirected_to={result.guardrail.redirected_block_id}")

# 4. Batch mode: generate a small synthetic training dataset
gen_config = ScenarioGeneratorConfig(n_scenarios=10, random_seed=7)
config.max_ticks = 60  # keep smoke test fast
generator = ScenarioGenerator(config, gen_config)
exporter = generator.run()
df = exporter.to_dataframe()
print(f"\nBatch generation produced {len(df)} telemetry rows across {gen_config.n_scenarios} scenarios")
print(df[["scenario_id", "tick", "block_id", "speed_kmh", "signal_aspect", "anomaly_active"]].head(8))

exporter.to_parquet("output/sample_training_data.parquet")
exporter.to_csv("output/sample_training_data.csv")
print("\nWrote output/sample_training_data.parquet and .csv")
