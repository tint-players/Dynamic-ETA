# Dynamic ETA — Railway Telemetry Simulator

This repository contains the Phase-2 railway simulation stack used to generate realistic telemetry for dynamic ETA modelling. It includes the simulator core, restrictive multi-train network engine, FastAPI/WebSocket backend, React dashboard, randomized scenario generation, and Parquet/CSV exporters.

## Current operational model

The corridor has shared logical block geometry and track-specific operational sections.

For the Delhi–Agra scenario there are 8 logical blocks and 2 physical tracks, producing 16 canonical track-block identities:

```text
UP-BLK-01 ... UP-BLK-08
DOWN-BLK-01 ... DOWN-BLK-08
```

A logical block such as `BLK-05` owns shared geometry such as length, gradient and curve information. Operational state that can differ between parallel tracks uses the pair `(track_id, block_id)` and the stable `track_block_id`.

Scope rules:

- shared logical-block state: weather and route geometry;
- track-specific state: occupancy, signals, TSRs, maintenance restrictions, platform state and traffic;
- explicit multi-track infrastructure: level crossings and crossovers.

The baseline Delhi–Agra YAML remains CLEAR weather with no TSR or maintenance restriction.

## Important simulator paths

```text
simulator/
  models.py                         Pydantic domain/config/telemetry models
  track_blocks.py                   canonical track-block identity helpers
  track_block_validation.py         track-aware configuration validation
  network_engine.py                 multi-train network foundation
  network_engine_v4.py              crossover, body occupancy and crossing safety
  network_engine_v4_restrictive.py  restrictive signalling, station occupancy,
                                    maintenance closures and manual signals
  exporters.py                      established telemetry/outcome aggregation
  track_aware_exporters.py          canonical track-block telemetry/visit adapters
  scenario_generator.py             randomized batch scenario generation
  dataset.py                        post-run ground-truth labelling
backend/
  session.py                        isolated live sessions and constraint injection
  main.py                           FastAPI/WebSocket API
  viz.py                            canonical visualization contract
frontend/                           React + TypeScript simulator dashboard
examples/delhi_agra_corridor.yaml   main Phase-2 corridor
smoke_test.py                       architecture-level end-to-end check
```

## Engine usage

`NetworkSimulationEngineV4Restrictive` is the current engine for the multi-track Delhi–Agra simulator, live dashboard sessions and multi-track scenario generation.

`SimulationEngine` is retained as a legacy single-train compatibility/regression engine. It must not be used as the authoritative implementation for multi-train signalling, crossover or track-specific operational behaviour.

## Dataset identity and leakage rule

Tick telemetry preserves both:

- `block_id` — shared logical/geographic block;
- `track_block_id` — canonical operational section.

A crossover occurring inside one logical block is therefore exported as separate track-block visits instead of one merged visit.

Ground-truth ETA fields (`actual_remaining_time_s`, arrival time and total journey time) are generated only after a completed run. They are labels, not live model inputs.

## Randomized generation

```python
from simulator import load_simulation_config, ScenarioGenerator, ScenarioGeneratorConfig

config = load_simulation_config("examples/delhi_agra_corridor.yaml")
generator = ScenarioGenerator(config, ScenarioGeneratorConfig(n_scenarios=100))
exporter = generator.run()
exporter.to_parquet("output/training_data.parquet")
```

On a multi-track route, generated TSRs are assigned to a specific canonical track-block. Weather remains logical-block scoped.

## Validation

From the repository root:

```bash
pip install -r backend/requirements.txt
python smoke_test.py
pytest backend/tests -q

cd frontend
npm install
npm run typecheck
npm run build
```

The test suite includes signalling/crossover regressions, turnaround behaviour, station occupancy, full maintenance closures, track-specific restrictions, canonical track-block export, visualization contracts and randomized scenario generation.

## Next modelling stage

The intended first ETA model is a hybrid temporal + graph model using the 16 canonical track-blocks as graph nodes. Model-visible state must remain observable operational state only; future outcome fields and hidden future disruption clearing times are not features.
