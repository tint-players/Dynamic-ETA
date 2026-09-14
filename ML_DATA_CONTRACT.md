# ML Dataset Contract

This document defines the minimum contract between the simulator and the ETA model pipeline.

## Prediction cadence

The simulator emits telemetry every simulation tick. The current Delhi–Agra configuration uses a 1-second tick, so the ML pipeline may create one prediction sample per active train per second. The planned LSTM look-back window is 60 seconds; that window length does not change the 1-second prediction cadence.

For the first 59 seconds of a train history, the shared feature builder left-pads the missing history with zero rows. The `history_present` sequence feature is `0` for padded rows and `1` for real observations. This allows prediction/training samples to exist from the first observable second while preserving a fixed `[60, F_sequence]` tensor shape.

## Provenance

Every row admitted to an ML training dataset must include:

- `run_id`: globally unique identifier for one simulation run. Dataset splitting must group by this field.
- `scenario_id`: human-readable scenario identity within a generated batch.
- `random_seed`: seed used by the scenario generator. Together with scenario identity it records the randomization stream used to construct the run.
- `train_id`: train whose state the row describes.
- `sim_time_s`: simulation time of the observation.
- `track_id` and `current_block_id`: raw operational location components.
- `track_block_id`: canonical operational section derived by the track-aware exporter, for example `UP-BLK-05`.

`run_id` is deliberately separate from `scenario_id`. Re-running scenario generation can produce another `scenario_00000`; the run identifier must still remain globally distinct so trajectories from different generator invocations cannot collide.

## Ground-truth label

The primary ETA target is:

```text
actual_remaining_time_s = actual_arrival_simulation_s - sim_time_s
```

The value is calculated only after the train has completed its journey. It is a training/evaluation label, never a live feature.

The following post-run values are forbidden from inference-time model inputs:

- `actual_remaining_time_s`
- `actual_arrival_simulation_s`
- `total_journey_time_s`

The shared feature builder intentionally does not expose a target field. Label attachment belongs to the later training-sample layer, so the same feature code can be reused for offline training and live inference without a second implementation.

## Shared feature builder

`simulator.ml_features.MLFeatureBuilder` is the canonical feature path. It accepts `SimulationConfig`, target-train history, and all current telemetry frames for one simulation second. It returns pure Python numeric arrays and does not depend on pandas, PyTorch, or the backend/frontend.

The same code path supports:

```text
live simulator frames ─┐
                      ├─> MLFeatureBuilder ─> model inputs
Parquet/dataframe rows ─┘
```

`build_from_records()` converts offline row dictionaries back into the same `TelemetryFrame` contract and then calls the identical core builder. Extra exporter-only fields such as `track_block_id` do not alter the features because canonical location is recomputed from `track_id + current_block_id`.

### Sequence features

The sequence tensor has a fixed 60-step window. V1 contains current train motion, normalized local/route progress, effective/current speed limits, gradient/curve state, weather and visibility, first/second signal look-ahead, speed/curve/TSR/crossing look-ahead, plus stable categorical flags for control action and broad control-reason classes. Optional values use zero plus an explicit presence mask.

Post-run label fields are ignored by construction. Tests assert that changing those values cannot change any live feature tensor.

### Graph node features

Nodes are canonical operational track-blocks in the stable order returned by `enumerate_track_blocks(route)`. For Delhi–Agra this produces 16 nodes.

V1 graph features combine:

- shared block geometry: relative length, speed limit, gradient, curve radius/limit;
- operational identity: track index, station/signal presence, crossing presence, crossover endpoint flags;
- current traffic: train-body occupancy, train count, mean/min speed, stopped-train count;
- current shared weather and visibility;
- active track-specific TSR and maintenance state;
- current dwelling-platform occupancy.

Occupancy uses train-body overlap rather than only the train front. While a train front is physically inside a crossover, both connected tracks are treated as occupied for graph-state construction, matching the simulator's multi-track crossover semantics.

Signal *presence* is included as a static node attribute. A global per-node dynamic signal-aspect feature is intentionally not reconstructed independently in the feature builder because the restrictive engine has subject-train exclusion, manual restrictive overrides, and Agra-specific signalling rules. The target train's observable next/second signal aspects are already present in the sequence features. If the GNN later needs global live signal aspects, those states should first be exported explicitly from the engine/API and then consumed here rather than duplicating signalling logic.

Similarly, crossing presence is a graph attribute while the target train's currently observable next-crossing state remains in the sequence. Dynamic V4 crossing phases are engine runtime state and should be exported explicitly before becoming a global node feature.

### Context features

The context vector contains normalized train capabilities (maximum speed, length, acceleration, service/emergency deceleration), current direction, active/station flags, and current track index. `current_node_index` is returned separately and points directly into the canonical node ordering.

## Dataset splitting

Rows must never be randomly split independently. All rows sharing a `run_id` belong to exactly one of train, validation, or test. This prevents adjacent one-second observations from the same simulated trajectory from leaking across splits.

A later checkpoint will implement the split utility; the required grouping key is fixed here as `run_id`.

## Planned model sample

For each eligible prediction second, the later sample builder will emit:

```text
X_seq              [60, F_sequence]
X_graph            [N_track_blocks, F_node]
edge_index         [2, E]
current_node_index scalar
route_mask         [N_track_blocks]
X_context          [F_context]
y                  actual_remaining_time_s
```

Checkpoint 2 now provides `X_seq`, `X_graph`, `X_context`, the canonical `node_ids`, and `current_node_index`. Graph edges and route masks belong to the next graph/sample checkpoints.

For the current Delhi–Agra corridor, `N_track_blocks = 16`. The same contract is intended to support larger future networks without changing the identity semantics.
