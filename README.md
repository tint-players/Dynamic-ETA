# Dynamic ETA — Railway Telemetry & ETA Prediction

Dynamic ETA combines a multi-train railway simulator, synthetic telemetry generation, machine-learning ETA modelling, live backend inference and a React dashboard.

The current repository models the Delhi–Agra corridor with 8 shared logical blocks across 2 physical tracks, producing 16 canonical operational track-blocks. The simulator generates one-second operational telemetry that can be labelled after completed journeys and transformed into temporal, graph and train-context inputs for ETA modelling.

## Documentation

Use these documents as the source of truth instead of duplicating architecture/path information across multiple root Markdown files:

- **[Project Overview](docs/OVERVIEW.md)** — problem scope, end-to-end architecture, simulator/ML/backend relationship and current status.
- **[Simulator & Data Generation](docs/SIMULATOR.md)** — railway representation, engine behaviour, telemetry, randomized scenarios and ground-truth generation.
- **[Machine Learning](docs/ML.md)** — dataset contract, leakage prevention, feature engineering, railway graph, model experiments and live ETA packaging.
- **[File Structure](docs/FILE_STRUCTURE.md)** — dedicated repository path/file-location reference.

## End-to-end flow

```text
Delhi–Agra configuration
        ↓
Railway simulator
        ↓
1-second telemetry
        ↓
Randomized / completed runs
        ↓
Post-run ground-truth labels
        ↓
Feature engineering
  ┌─────┼─────┐
  ↓     ↓     ↓
60 s   graph  train
seq.   state  context
  └─────┼─────┘
        ↓
ETA model
        ↓
Live backend / dashboard
```

## Current operational model

The canonical track-block identities are:

```text
UP-BLK-01 ... UP-BLK-08
DOWN-BLK-01 ... DOWN-BLK-08
```

Route geometry and weather are shared at logical-block level. Occupancy, signals, TSRs, maintenance, platform state and traffic are track-specific. Crossings and crossovers explicitly identify the tracks they affect.

`NetworkSimulationEngineV4Restrictive` is the authoritative multi-track engine. The legacy `SimulationEngine` is retained only for compatibility/regression and is not the source of truth for multi-train signalling, crossovers or track-specific operational behaviour.

## Quick start

### Simulator/backend

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python smoke_test.py
pytest backend/tests -q
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server normally opens the dashboard at `http://localhost:5173`.

### Live ML ETA checkpoint

For the packaged Phase-3 live model:

```bash
pip install -r requirements.txt
python scripts/setup_live_eta.py
```

Automatic checkpoint retrieval requires an authenticated GitHub CLI. A manually downloaded checkpoint can instead be installed with:

```bash
python scripts/setup_live_eta.py --source /path/to/eta_winner.pt
```

See [Machine Learning](docs/ML.md) for the checkpoint contract and live-inference details.

## Dataset safety rule

Ground-truth outcome fields are generated only after a completed journey. In particular:

```text
actual_remaining_time_s
actual_arrival_simulation_s
total_journey_time_s
```

are labels/outcomes and must not be used as live model inputs.

## Current ML scope

The packaged Phase-3 model predicts remaining time to the final journey destination. The broader project goal includes ETA for upcoming/intermediate stations, but station-wise ETA requires a new target/label design and retraining; it is not an existing capability of the current checkpoint.

## Validation

A standard validation pass is:

```bash
python smoke_test.py
pytest backend/tests -q

cd frontend
npm run typecheck
npm run build
```

For detailed behaviour, contracts and implementation notes, follow the documentation links above. File locations are maintained only in [docs/FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md).