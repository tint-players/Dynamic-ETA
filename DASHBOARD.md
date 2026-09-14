# Simulator Dashboard

The dashboard is a thin FastAPI/WebSocket + React/TypeScript debugging layer over the simulator. The browser never calculates train physics. Playback speed only changes wall-clock emission rate; it does not change `simulation.tick_seconds`.

For the Delhi–Agra multi-track scenario, live sessions use `NetworkSimulationEngineV4Restrictive`. The legacy single-train `SimulationEngine` remains only for compatibility/regression checks and is not the source of truth for multi-train signalling, crossovers or track-specific restrictions.

## Track-block visualization contract

The backend exposes both shared logical blocks and canonical operational track-blocks.

```text
BLK-01 ... BLK-08                  shared route geometry
UP-BLK-01 ... UP-BLK-08            TRACK-UP operational sections
DOWN-BLK-01 ... DOWN-BLK-08        TRACK-DOWN operational sections
```

Weather remains logical-block scoped. Signals, TSRs, maintenance, platform occupancy and traffic are track-specific. Crossings and crossovers explicitly reference the tracks they affect.

## Run the backend

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

## Run the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

## REST API

- `GET /api/health` — health check.
- `GET /api/scenarios` — list YAML scenarios from `examples/`.
- `POST /api/sessions` — create an isolated simulator session.
- `GET /api/sessions/{session_id}/config` — canonical route/infrastructure visualization contract, including track-blocks.
- `POST /api/sessions/{session_id}/reset` — reconstruct the original session configuration and simulator state.

Only known YAML basenames in `examples/` can be loaded; arbitrary browser-supplied filesystem paths are rejected.

## Live constraints

Manual TSR and maintenance injections on a multi-track route require an explicit `track_id`. Active restriction state also includes the derived canonical `track_block_id`.

Manual railway signal overrides are restrictive-only: RED or YELLOW. Clearing an override returns that signal to automatic dynamic signalling.

## WebSocket behaviour

The existing client commands include play, pause, step, reset and playback-speed control, plus the supported live constraint injection/reset commands exposed by the backend.

Telemetry is emitted per train. `actual_remaining_time_s` is intentionally unavailable during a genuinely live run because it is post-journey ground truth, not a prediction or live input.

## Validation

From the repository root:

```bash
python smoke_test.py
pytest backend/tests -q

cd frontend
npm run typecheck
npm run build
```

The backend regression suite covers route geometry, session reset, dynamic signalling, restrictive-signal approach, train separation, crossover reservation/turnaround, station occupancy, maintenance closures, track-specific restrictions, track-block export, visualization contracts and API behaviour.
