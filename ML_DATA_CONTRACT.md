# ML Dataset Contract

This document defines the minimum contract between the simulator and the ETA model pipeline.

## Prediction cadence

The simulator emits telemetry every simulation tick. The current Delhi–Agra configuration uses a 1-second tick, so the ML pipeline may create one prediction sample per active train per second. The planned LSTM look-back window is 60 seconds; that window length does not change the 1-second prediction cadence.

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

## Dataset splitting

Rows must never be randomly split independently. All rows sharing a `run_id` belong to exactly one of train, validation, or test. This prevents adjacent one-second observations from the same simulated trajectory from leaking across splits.

A later checkpoint will implement the split utility; the required grouping key is fixed here as `run_id`.

## Planned model sample

For each eligible prediction second, the feature builder will eventually emit:

```text
X_seq              [60, F_sequence]
X_graph            [N_track_blocks, F_node]
edge_index         [2, E]
current_node_index scalar
route_mask         [N_track_blocks]
X_context          [F_context]
y                  actual_remaining_time_s
```

For the current Delhi–Agra corridor, `N_track_blocks = 16`. The same contract is intended to support larger future networks without changing the identity semantics.
