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

The shared feature builder intentionally does not expose a target field. Label attachment belongs to the training-sample layer, so the same feature code is reused for offline training and live inference without a second implementation.

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

### Graph topology

`simulator.ml_graph.MLGraphTopologyBuilder` is the canonical topology source. It uses the exact same canonical track-block node order as the feature builder.

The operational graph preserves railway travel direction. For the current Delhi–Agra corridor it is:

```text
UP-BLK-01 -> UP-BLK-02 -> ... -> UP-BLK-08

DOWN-BLK-08 -> DOWN-BLK-07 -> ... -> DOWN-BLK-01

UP-BLK-07 -> DOWN-BLK-07   (XOVER-AGRA-01)
```

This gives 14 route-successor edges plus one directed crossover edge. Track direction is taken from the configured directional signals; a track with ambiguous signalling is rejected rather than silently assigned the wrong direction.

The topology exposes two edge views:

- `operational_edge_index`: authoritative directed railway movement edges, with `ROUTE_SUCCESSOR` or `CROSSOVER` edge type metadata;
- `edge_index`: a symmetric message-passing view containing both orientations of each operational edge, intended for the initial GraphSAGE encoder.

Keeping both views means the model can aggregate context from both neighbours without losing the true directed railway topology.

The builder also provides a leakage-safe `route_mask(train_id)`. The mask is derived only from configuration: the source-to-destination block span on the train's starting track plus both endpoints of any explicitly configured crossover. It does not inspect future runtime outcomes or arrival labels.

### Context features

The context vector contains normalized train capabilities (maximum speed, length, acceleration, service/emergency deceleration), current direction, active/station flags, and current track index. `current_node_index` is returned separately and points directly into the canonical node ordering.

## Per-second sample builder

`simulator.ml_samples.MLSampleBuilder` is the boundary between simulator state and the eventual PyTorch dataset/model code.

`build_inputs()` is the live-safe path. It packages the exact feature tensors and topology needed for one ETA prediction second and cannot access a target label:

```text
MLModelInputs
  x_seq                   [60, F_sequence]
  x_graph                 [N_track_blocks, F_node]
  edge_index              [2, E_message]
  operational_edge_index  [2, E_operational]
  operational_edge_types
  current_node_index
  route_mask              [N_track_blocks]
  x_context               [F_context]
  node_ids
```

`build_training_sample()` is offline-only. It first builds the same `MLModelInputs`, then attaches `y = actual_remaining_time_s` plus `run_id`, `scenario_id`, `random_seed`, `train_id`, and `sim_time_s`. Missing provenance or a missing/negative target is rejected.

Both paths also have record adapters for exported Parquet/dataframe-style dictionaries. Tests require record-based and live-frame inference inputs to be exactly equal. Therefore future training and live serving code consume the same model-input contract rather than maintaining separate preprocessing implementations.

## Training dataset construction

`simulator.ml_dataset.MLTrainingDatasetBuilder` converts one completed, labelled simulation run into the actual sequence of training examples. It groups telemetry by simulation second, passes all trains at that second to the shared sample builder, and by default emits one `MLTrainingSample` for every train that is currently active and not yet completed.

This deliberately excludes pre-departure inactive trains and post-arrival completed rows. Stopped trains, station dwell, signal waits, crossing waits, and other active zero-speed states remain valid prediction samples.

The builder requires exactly one `run_id` at a time and rejects a run that mixes scenario identities or random seeds. This keeps each constructed trajectory internally consistent.

The default sample cadence remains every simulator step. `sample_every_n_steps` is an optional memory-control knob for pilot experiments. When it is greater than one, only target samples are thinned; every one-second telemetry frame is still appended to history before the retained target is built. Therefore a retained sample continues to use the same 60-second, 1-second-resolution temporal context as live inference. This option is not a change to the production prediction cadence.

The resulting default cadence is therefore:

```text
run 1 / train A / t=0     -> sample
run 1 / train A / t=1     -> sample
run 1 / train B / t=1     -> sample if active
...
run 1 / train A / arrival -> no later completed samples
```

## Randomized scenario retention

`ScenarioGenerator.generate_runs()` returns `GeneratedScenarioRun` objects rather than discarding the scenario configuration after simulation. Each object contains the globally unique `run_id`, scenario identity, generator seed, the exact randomized `SimulationConfig`, and the completed labelled frames.

This exact-config retention is required for offline graph feature construction. Weather, TSR, maintenance, crossings, and signal schedules can differ between generated runs; rebuilding graph features with the unchanged base YAML would silently describe the wrong operational state. `ScenarioGenerator.run()` remains as a backward-compatible tabular export path and is implemented on top of the retained generated runs.

Non-baseline randomized scenarios currently cover shared weather, track-specific TSR, crossing closures, signal restrictions, and track-specific maintenance. Maintenance randomization includes both speed restrictions and finite-duration full closures. Generated full closures always have an end time so scenario generation does not intentionally create permanent deadlocks.

## Dataset splitting

`split_training_samples_by_run()` partitions complete runs, never individual one-second rows. The default ratio is 70% train, 15% validation, 15% test, with deterministic shuffling controlled by `split_seed`.

All samples with the same `run_id` are guaranteed to remain in exactly one partition. For small datasets, when there are at least three runs and all three ratios are non-zero, the splitter reserves at least one run for each partition before distributing the remainder. This avoids accidentally evaluating on an empty validation or test set during early experiments.

The split object records both the samples and the exact `run_id` membership for train, validation, and test, making leakage audits straightforward.

## Model benchmarks

The model stack is built as explicit ablations so each added source of information has to earn its complexity.

The LSTM-only benchmark consumes `x_seq + x_context`. Its default encoder is a two-layer LSTM with hidden size 128 and a non-negative ETA regression head.

The graph-only benchmark in `simulator.ml_gnn` consumes `x_graph + edge_index + current_node_index + route_mask + x_context`. The default graph encoder is three mean-aggregation GraphSAGE layers with hidden size 128. It produces one embedding per canonical track-block. The ETA head combines the embedding of the train's current track-block, a masked mean embedding over the train's configured route, and train context, then uses `Softplus` to keep ETA non-negative.

`edge_index` is used for GraphSAGE message passing. `operational_edge_index` and edge-type metadata remain outside this first graph benchmark so true directed railway semantics are preserved for later directional/relational GNN experiments without changing the data contract.

The intended evaluation ladder is therefore:

```text
naive ETA baselines
        ↓
LSTM-only
        ↓
GraphSAGE-only
        ↓
LSTM + GraphSAGE hybrid
```

All neural benchmarks use the same post-run `actual_remaining_time_s` target and Huber regression loss, and they are evaluated only on run-level validation/test partitions.

## Training experiment protocol

`simulator.ml_experiment.run_eta_experiment()` is the canonical ablation experiment runner. All candidate models use one frozen run-level split and therefore see the same train, validation, and held-out test runs.

The three deterministic baselines are evaluated on validation. LSTM-only, GraphSAGE-only, and hybrid models are trained with AdamW and Huber loss, using validation MAE for early stopping/checkpoint selection. Candidate selection is performed using validation MAE only. The held-out test split is not used to select an architecture, tune a checkpoint, or choose an epoch; after the validation winner is fixed, only that winner is evaluated on test.

The command-line entry point is `train_eta_experiment.py`. It records a JSON summary of run/sample membership and candidate validation results. `save_winner_checkpoint()` stores the selected model identity, constructor arguments, neural state dict when applicable, experiment settings, validation/test metrics, exact run IDs for each split, and the sequence/graph/context feature schema. This metadata is part of the audit trail required before a checkpoint is used for live inference.

A small `sample_every_n_steps > 1` value may be used for CI or memory-constrained pilot runs, but any reported experiment must record that value. A statistically meaningful final experiment should use the intended full cadence when hardware/memory allows, or explicitly disclose a different target-sampling cadence.

## Current model sample

For each eligible prediction second the sample layer now provides:

```text
X_seq                   [60, F_sequence]
X_graph                 [N_track_blocks, F_node]
edge_index              [2, E_message]
operational_edge_index  [2, E_operational]
current_node_index      scalar
route_mask              [N_track_blocks]
X_context               [F_context]
y                       actual_remaining_time_s  # offline training only
```

For the current Delhi–Agra corridor, `N_track_blocks = 16`. The same contract is intended to support larger future networks without changing the identity semantics.
