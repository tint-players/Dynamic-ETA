# Railway Simulator & Data Generation

This document describes the simulator and synthetic-data side of Dynamic ETA. For exact source locations, see [File Structure](FILE_STRUCTURE.md).

## 1. Simulation purpose

The simulator provides controlled railway telemetry for ETA research when a sufficiently rich real operational feed is unavailable. It models train motion together with operational constraints so that generated delays arise from railway conditions rather than arbitrary ETA noise.

The Delhi–Agra corridor is the main scenario. It uses 8 logical blocks and 2 physical tracks, producing 16 canonical operational track-block identities.

## 2. Network representation

A logical block represents shared geography. Length, gradient, curve information and weather belong to this shared geographic layer.

A canonical track-block represents one operational section on one physical track. Occupancy, signals, TSRs, maintenance, platform state and traffic can therefore differ between UP and DOWN even when both tracks pass through the same logical block.

Canonical identities follow the stable form:

```text
UP-BLK-01 ... UP-BLK-08
DOWN-BLK-01 ... DOWN-BLK-08
```

Track-aware validation rejects ambiguous or invalid operational references instead of silently merging parallel-track state.

## 3. Authoritative engine

`NetworkSimulationEngineV4Restrictive` is the authoritative engine for the multi-track Delhi–Agra simulator, live dashboard sessions and multi-track randomized generation.

The older `SimulationEngine` remains for legacy single-train compatibility/regression. It is not the source of truth for multi-train signalling, crossovers or track-specific operational behaviour.

The restrictive network engine combines:

- multi-train movement and occupancy;
- train-body rather than front-point occupancy where operationally required;
- dynamic restrictive signalling;
- station and platform occupancy;
- level-crossing safety;
- crossover reservation and track transitions;
- track-specific TSR and maintenance;
- manual restrictive signal overrides.

## 4. Discrete-time working procedure

The current corridor runs at one-second simulation ticks. Conceptually each tick performs the following cycle:

```text
current train/network state
          ↓
observe constraints ahead
          ↓
determine governing speed/stop target
          ↓
controller chooses motion response
          ↓
update acceleration, speed and position
          ↓
update occupancy/infrastructure state
          ↓
emit per-train telemetry
          ↓
advance to next simulation second
```

Playback speed in the dashboard affects only wall-clock delivery. It does not alter this simulation-time resolution.

## 5. Train movement and effective restrictions

A train's motion is constrained by its physical capabilities and the railway state around it. Relevant limits include train maximum speed, block speed limit, curves, temporary restrictions and signalling/stop constraints. The governing controller responds by accelerating, maintaining speed, braking, stopping or waiting as appropriate.

The important modelling idea is that operational delay emerges from the simulated constraints. A red signal, occupied platform, crossing closure, crossover conflict, TSR or maintenance closure changes the train trajectory, which then changes the resulting arrival time.

## 6. Signalling

Signals are dynamic and track-specific. The restrictive engine evaluates occupancy and operational rules to determine governing signal behaviour. Restrictive YELLOW/RED conditions produce approach or stop behaviour rather than being treated as labels only.

Manual signal overrides are intentionally restrictive-only: RED and YELLOW may be imposed. Clearing the override returns the signal to automatic dynamic operation.

## 7. Body occupancy and multi-train interaction

A train has physical length. Operational occupancy therefore cannot always be represented by the train front alone. Train-body overlap is used for relevant block, platform, crossover and ML graph occupancy calculations.

This matters near boundaries because a train's front may have entered a new section while its rear still occupies the previous protected area.

## 8. Stations and platforms

Scheduled stops include approach, braking, platform stop, dwell and departure. Platform occupancy is part of the track-specific operational state. If the required platform area is occupied, the restrictive engine can hold an approaching train before the occupied area instead of allowing physical overlap.

## 9. Crossings and crossovers

Level crossings are explicit infrastructure affecting the tracks they protect. A closure can create a stop/wait condition and therefore generate realistic operational delay.

Crossovers connect tracks and require conflict-safe reservation. The Agra configuration contains the operational crossover connection used by the turnaround/cross-track scenario. Crossover occupancy and reservation semantics are preserved by the V4/restrictive engine.

## 10. TSR and maintenance

A TSR is a Temporary Speed Restriction. In the multi-track model it is assigned to a specific `(track_id, block_id)` / canonical track-block so that a restriction on one track does not automatically restrict the parallel track.

Maintenance is also track-specific and may impose a reduced speed or a finite full closure. Randomly generated full closures have an end time so generation does not intentionally create permanent deadlocks.

## 11. Weather

Weather is shared at logical-block/geographic level rather than duplicated independently for each parallel track. Randomized runs may alter weather and visibility while preserving this scope rule.

## 12. Telemetry contract

The simulator emits a telemetry observation for each train at each simulation tick. Telemetry carries the train's current motion/location and observable operational context. Track-aware export preserves both:

- `block_id` — shared logical/geographic block;
- `track_block_id` — canonical operational section.

This prevents a crossover or parallel-track visit inside the same logical block from being collapsed into one ambiguous location.

## 13. Randomized scenario generation

Synthetic training data is created by repeatedly running the base corridor under controlled randomization. Non-baseline generated scenarios can vary:

- shared weather/visibility;
- track-specific TSR;
- crossing closures;
- restrictive signal schedules;
- track-specific maintenance, including finite full closures.

The baseline corridor remains CLEAR with no TSR or maintenance restriction.

Each retained generated run stores a globally unique `run_id`, scenario identity, random seed, exact randomized simulation configuration and completed labelled frames. Keeping the exact randomized configuration is essential because graph/operational features must describe the scenario that actually ran, not the unchanged base YAML.

## 14. Ground-truth generation

True ETA labels are generated only after a train completes its journey. The current destination target is:

```text
actual_remaining_time_s = actual_arrival_simulation_s - sim_time_s
```

The post-run fields are outcome labels and are forbidden as live model inputs:

```text
actual_remaining_time_s
actual_arrival_simulation_s
total_journey_time_s
```

This separation prevents target leakage.

## 15. Dataset generation example

A typical randomized export uses the scenario loader and generator, runs multiple scenarios and writes the resulting labelled telemetry to a tabular format such as Parquet.

```python
from simulator import load_simulation_config, ScenarioGenerator, ScenarioGeneratorConfig

config = load_simulation_config("examples/delhi_agra_corridor.yaml")
generator = ScenarioGenerator(config, ScenarioGeneratorConfig(n_scenarios=100))
exporter = generator.run()
exporter.to_parquet("output/training_data.parquet")
```

## 16. Backend/dashboard simulator interface

The backend creates isolated simulator sessions and exposes the canonical route/infrastructure visualization contract. The REST layer supports health, scenario listing, session creation/configuration and reset. WebSocket commands support playback and the live constraint operations implemented by the backend.

On multi-track routes, manual TSR and maintenance injections require an explicit track. The browser visualizes simulator state but does not calculate train physics.

## 17. Validation

From the repository root, simulator/backend validation includes the smoke test and backend pytest suite. Frontend validation uses TypeScript checking and the production build. The regression suite covers route geometry, signalling, restrictive approaches, train separation, crossover reservation/turnaround, station occupancy, maintenance closures, track-specific restrictions, track-block export, visualization contracts, API behaviour and randomized generation.

## 18. Relationship to ML

The simulator is the data source; it is not the ETA predictor. Completed runs provide labelled trajectories, while live frames provide inference-safe observations. The ML layer converts these observations into temporal, graph and train-context features.

Continue with [Machine Learning](ML.md).