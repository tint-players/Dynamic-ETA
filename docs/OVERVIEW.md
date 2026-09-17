# Dynamic ETA — Project Overview

## Purpose

Dynamic ETA is a railway telemetry simulation and machine-learning project for estimating train arrival time from changing operational conditions rather than relying only on a static timetable or current delay.

The repository currently combines four layers:

1. a multi-train railway simulator for the Delhi–Agra corridor;
2. synthetic scenario and labelled dataset generation;
3. an ETA modelling stack using temporal and railway-network features;
4. a FastAPI/WebSocket backend and React dashboard for live simulation and ETA display.

For the complete repository path map, see [File Structure](FILE_STRUCTURE.md).

## End-to-end architecture

```text
Delhi–Agra scenario configuration
            ↓
Railway simulation
            ↓
1-second per-train telemetry
            ↓
Randomized scenario runs
            ↓
Post-run ground-truth labelling
            ↓
Shared ML feature construction
      ┌─────┼─────┐
      ↓     ↓     ↓
  sequence graph context
      └─────┼─────┘
            ↓
ETA model
            ↓
Live backend inference
            ↓
WebSocket / dashboard
```

## Operational railway model

The Delhi–Agra scenario contains 8 shared logical/geographic blocks and 2 physical tracks, producing 16 canonical operational track-blocks:

```text
UP-BLK-01 ... UP-BLK-08
DOWN-BLK-01 ... DOWN-BLK-08
```

A logical block owns shared route geometry such as length, gradient and curve information. Operational state that may differ between parallel tracks is identified by `(track_id, block_id)` and the stable `track_block_id`.

Scope rules are:

- shared logical-block state: route geometry and weather;
- track-specific state: occupancy, signalling, TSR, maintenance, platform state and traffic;
- explicit multi-track infrastructure: level crossings and crossovers.

The baseline Delhi–Agra YAML uses CLEAR weather and no TSR or maintenance restriction.

## Simulator and synthetic data

The authoritative multi-track simulator is `NetworkSimulationEngineV4Restrictive`. It builds on the network engine with restrictive signalling, station/platform occupancy, maintenance closures and manual restrictive signals while retaining crossing, crossover and train-body occupancy behaviour.

The simulator advances in discrete ticks and the current Delhi–Agra configuration emits telemetry at one-second resolution. Randomized scenario generation can vary weather, track-specific TSRs, crossing closures, restrictive signals and track-specific maintenance. Completed runs are labelled only after the actual journey outcome is known.

See [Simulator & Data Generation](SIMULATOR.md) for the detailed simulator workflow and data-generation contract.

## ML pipeline

The ML pipeline separates inference-safe inputs from post-run labels. Its shared feature path represents each eligible prediction instant with:

- a 60-step temporal sequence;
- a 16-node canonical track-block graph for the current corridor;
- railway graph topology and route mask;
- train-specific context.

The experiment stack contains deterministic ETA baselines, LSTM-only, GraphSAGE-only and LSTM + GraphSAGE hybrid models. Candidate selection is performed on validation runs and the selected model is then evaluated on held-out test runs.

The currently packaged live checkpoint predicts remaining time to the train's final journey destination. Intermediate/station-wise ETA is not implemented by the current target contract and should be treated as a future modelling extension rather than an existing capability.

See [Machine Learning](ML.md) for the complete data contract, feature construction, topology, training and live-inference design.

## Backend and dashboard

The dashboard is a thin visualization and control layer over the simulator. Train physics is calculated by the backend simulator, not by the browser. Playback speed changes wall-clock emission speed; it does not change the simulator tick duration.

The backend exposes REST endpoints for health, scenario/session management and visualization configuration, plus WebSocket commands for playback and supported live constraint operations. Multi-track live sessions use the restrictive network engine.

The visualization contract exposes both shared logical blocks and canonical track-blocks. Weather remains shared at logical-block level, while signals, TSRs, maintenance, platform occupancy and traffic are track-specific. Crossings and crossovers explicitly identify affected tracks.

Manual multi-track TSR and maintenance injections require a `track_id`. Manual signal overrides are restrictive-only (RED or YELLOW); clearing an override restores automatic dynamic signalling.

Live telemetry deliberately does not expose post-run ground-truth remaining time as a prediction input.

## Current live ETA packaging

The selected live checkpoint is stored outside normal source control and is installed at runtime. The packaged setup script retrieves and verifies the selected experiment artifact before use. If PyTorch or the checkpoint is unavailable or invalid, the simulator remains usable and live ML ETA is reported as unavailable instead of preventing simulator startup.

The live path reuses the same model-input builder used by offline training, preventing a separate frontend/backend preprocessing implementation from drifting away from the training feature contract.

## Validation

The project uses simulator smoke tests, backend regression tests and frontend TypeScript/build validation. Coverage includes signalling and crossover regressions, turnaround behaviour, station occupancy, maintenance closures, track-specific restrictions, canonical track-block export, visualization contracts, scenario generation and Phase-3 ML/live-ETA components.

## Documentation

- [README](../README.md) — navigation and quick start.
- [Simulator & Data Generation](SIMULATOR.md) — simulator architecture, telemetry and randomized data generation.
- [Machine Learning](ML.md) — dataset contract, features, graph, models, experiments and live ETA.
- [File Structure](FILE_STRUCTURE.md) — repository path reference.