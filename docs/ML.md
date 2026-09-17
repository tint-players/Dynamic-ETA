# Machine Learning & ETA Prediction

This document consolidates the Phase-3 dataset contract, feature engineering, graph topology, model experiments and packaged live ETA path. For exact source locations, see [File Structure](FILE_STRUCTURE.md).

## 1. Prediction contract

The simulator emits telemetry every simulation tick. The Delhi–Agra configuration uses a one-second tick, so the ML pipeline can create one prediction sample per active train per second.

The current trained target is destination remaining time:

```text
actual_remaining_time_s = actual_arrival_simulation_s - sim_time_s
```

The label is calculated only after the train has completed its journey. The currently packaged model therefore predicts remaining time to the final journey destination. Intermediate/station-wise ETA requires a new target/label design and is not part of this current checkpoint contract.

## 2. Provenance

Every row admitted to the training pipeline records run and trajectory identity, including:

- `run_id` — globally unique simulation-run identity and the grouping key for dataset splitting;
- `scenario_id` — human-readable scenario identity;
- `random_seed` — generator randomization seed;
- `train_id` — target train;
- `sim_time_s` — observation time;
- `track_id`, `current_block_id` and canonical `track_block_id` — operational location.

`run_id` is separate from `scenario_id` so repeated generator invocations cannot accidentally merge trajectories with reused scenario names.

## 3. Leakage prevention

The following post-run outcome fields are forbidden from inference-time model inputs:

```text
actual_remaining_time_s
actual_arrival_simulation_s
total_journey_time_s
```

The shared feature builder has no target field. Label attachment occurs only in the offline training-sample layer. This makes it possible to reuse the same feature construction for training and live inference.

## 4. Shared feature path

The canonical feature builder accepts the simulation configuration, the target train's history and all current telemetry frames for one simulation second. It produces pure numeric model inputs and is independent of pandas, PyTorch and the frontend/backend serving layer.

Both live simulator frames and exported records converge on the same core path:

```text
live TelemetryFrame history ─┐
                            ├─> shared feature builder ─> model inputs
exported/offline records ───┘
```

Record adapters reconstruct the telemetry contract and then call the same builder. Export-only fields do not silently create a separate preprocessing implementation.

## 5. Temporal sequence

The sequence input has a fixed 60-step window at one-second resolution. For the first 59 observable seconds, missing history is left-padded with zero rows and `history_present` distinguishes padding from real observations.

The temporal representation includes current train motion and route progress, current/effective speed constraints, gradient/curve information, weather/visibility, signal look-ahead, speed/curve/TSR/crossing look-ahead and stable control-action/control-reason categories.

Optional numeric values use zero together with an explicit presence mask so the model can distinguish “not present” from a real zero-valued observation.

## 6. Graph node features

Graph nodes are canonical operational track-blocks in stable track-aware order. For Delhi–Agra there are 16 nodes.

Static/infrastructure information includes shared block geometry, operational track identity and presence indicators for relevant station/signal/crossing/crossover infrastructure.

Dynamic graph state includes current train-body occupancy, train count, mean/minimum speed, stopped-train count, weather/visibility, active track-specific TSR and maintenance state, and dwelling-platform occupancy.

Occupancy uses train-body overlap rather than only the front position. While a train is physically within a crossover, the graph-state construction reflects the connected-track occupancy semantics used by the simulator.

Global dynamic signal aspects are intentionally not independently reconstructed in the graph feature builder because authoritative restrictive signalling contains subject-train exclusion, manual overrides and scenario-specific rules. The target train's observable next/second signal state is represented in the temporal sequence. Dynamic crossing phases follow the same principle: authoritative runtime state should be exported before being promoted to a global graph feature rather than duplicated by ML preprocessing.

## 7. Railway graph topology

The operational graph preserves travel direction:

```text
UP-BLK-01 -> UP-BLK-02 -> ... -> UP-BLK-08

DOWN-BLK-08 -> DOWN-BLK-07 -> ... -> DOWN-BLK-01

UP-BLK-07 -> DOWN-BLK-07   (Agra crossover)
```

This gives 14 route-successor edges plus one directed crossover edge.

Two edge views are retained:

- `operational_edge_index` — authoritative directed railway movement edges with edge-type metadata;
- `edge_index` — symmetric message-passing edges for the initial GraphSAGE encoder.

This lets the GNN aggregate neighbouring context in both directions without discarding the actual directed railway topology.

A leakage-safe `route_mask(train_id)` is derived from configured route information rather than future runtime outcomes.

## 8. Train context

The context vector describes target-train capabilities and current state, including normalized maximum speed, length, acceleration, service/emergency deceleration, direction, active/station flags and current track index. `current_node_index` separately identifies the target train's current canonical graph node.

## 9. Per-second model sample

The live-safe sample boundary packages:

```text
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

Offline training first builds these exact inputs and only then attaches:

```text
y = actual_remaining_time_s
```

plus provenance (`run_id`, `scenario_id`, `random_seed`, `train_id`, `sim_time_s`).

## 10. Training dataset construction

A completed labelled simulation run is grouped by simulation second. All trains at that second are available to the shared feature builder, and one sample is emitted for each train that is active and not completed.

This excludes pre-departure inactive rows and post-arrival completed rows. Active zero-speed states such as station dwell, signal waits, crossing waits and other operational holds remain valid samples.

An optional `sample_every_n_steps` setting can thin target samples for pilot/CI memory control without thinning the one-second history used to construct each retained temporal window.

## 11. Run-level dataset splitting

Dataset splitting occurs by complete `run_id`, never by adjacent telemetry rows. The default split is 70% train, 15% validation and 15% test with deterministic shuffling from a split seed.

Keeping complete runs in one partition prevents highly correlated seconds from the same journey leaking across train/validation/test boundaries. For sufficiently large small-run experiments, the splitter preserves non-empty positive partitions.

## 12. Model ladder

The Phase-3 model stack was built as an ablation ladder:

```text
naive ETA baselines
        ↓
LSTM-only
        ↓
GraphSAGE-only
        ↓
LSTM + GraphSAGE hybrid
```

The LSTM benchmark consumes temporal sequence plus train context. The GraphSAGE benchmark consumes graph state/topology, target-node information, route mask and train context. The hybrid combines temporal and graph representations before non-negative ETA regression.

Neural candidates use Huber regression loss. Candidate selection is based on validation MAE; the held-out test split is reserved until the validation winner is fixed.

## 13. Selected Phase-3 experiment

The packaged checkpoint was selected from GitHub Actions run `34871556440`, artifact `eta-experiment-20runs-12epochs-2s`.

Expected checkpoint SHA256:

```text
24feb1ca5cc30d84698a331324588e4305dcb4915dee77dc4680a11507b18655
```

The default runtime destination is:

```text
artifacts/eta_winner.pt
```

The checkpoint is not treated as ordinary source code. The setup script installs the selected artifact and verifies the fixed hash before live use.

## 14. Live ETA setup

Install the backend and ML dependencies used by this branch, then run:

```bash
pip install -r backend/requirements.txt
pip install -r requirements-ml.txt
python scripts/setup_live_eta.py
```

Automatic artifact retrieval uses GitHub CLI and therefore requires an authenticated `gh` session (or suitable token configuration). If the checkpoint has already been downloaded manually, install it with:

```bash
python scripts/setup_live_eta.py --source /path/to/eta_winner.pt
```

An alternate runtime checkpoint path may be supplied through:

```bash
export DYNAMIC_ETA_ML_CHECKPOINT=/absolute/path/to/eta_winner.pt
```

## 15. Live inference behaviour

PyTorch is intentionally optional for the core simulator startup. If PyTorch or the checkpoint is missing/invalid, simulation remains available and the dashboard reports ML ETA as unavailable.

For each live telemetry update, the backend retains one-second history, builds the same model inputs used offline, validates the checkpoint feature contract, performs inference for active incomplete trains and exposes predicted remaining time / predicted arrival simulation time through the live ETA state.

Post-run ground-truth fields are not read by the live inference path.

## 16. Evaluation terms

The main ETA error measures used by the experiment stack are absolute-error statistics:

- **MAE** — mean absolute prediction error;
- **Median absolute error** — the 50th-percentile absolute error;
- **P90 absolute error** — a tail metric below which 90% of absolute errors fall;
- **ETA buckets** — error broken down by remaining-time horizon.

Lower values indicate smaller prediction errors. These metrics measure model performance; they do not themselves change model accuracy.

## 17. Current limitation and next modelling question

The current Phase-3 checkpoint is trained on final-destination remaining time. The broader project goal includes dynamic arrival forecasts for upcoming stations, but that requires target-station-aware ground truth and model inputs/retraining. It should not be inferred from the existing destination target.

If the ML system is rebuilt from the Phase-2 simulator baseline, the first design decision should therefore be the exact prediction unit: next-station ETA, all upcoming-station ETAs, destination ETA, or a target-station-conditioned formulation.

## 18. Related documentation

- [Project Overview](OVERVIEW.md)
- [Simulator & Data Generation](SIMULATOR.md)
- [File Structure](FILE_STRUCTURE.md)
- [README](../README.md)