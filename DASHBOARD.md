# Simulator Dashboard

This branch adds a thin FastAPI/WebSocket adapter and a React + TypeScript debugging dashboard on top of the existing Component A simulator.

The Python `SimulationEngine` remains the source of truth. The browser does not calculate train physics. UI playback speed only changes how quickly backend ticks are emitted in wall-clock time; it does not change `simulation.tick_seconds`.

## Run the backend

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

The API is then available at `http://localhost:8000`.

## Run the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## REST API

- `GET /api/health` — basic health check.
- `GET /api/scenarios` — lists YAML scenarios available from `examples/`.
- `POST /api/sessions` — creates an isolated live simulator session. Body: `{ "scenario_name": "delhi_agra_corridor.yaml" }`.
- `GET /api/sessions/{session_id}/config` — returns canonical backend route/infrastructure geometry for visualization.
- `POST /api/sessions/{session_id}/reset` — reconstructs a clean `SimulationEngine` using the original config.

Arbitrary browser-supplied filesystem paths are not accepted. Only known YAML basenames in `examples/` can be loaded.

## WebSocket protocol

Connect to:

```text
ws://localhost:8000/ws/{session_id}
```

Client commands:

```json
{"command":"play"}
{"command":"pause"}
{"command":"step"}
{"command":"reset"}
{"command":"set_speed","speed":5}
```

Allowed playback rates are `0.5`, `1`, `2`, `5`, and `10`.

Server messages:

```json
{"type":"telemetry","frame":{}}
{"type":"playback_state","playing":true,"playback_speed":5,"complete":false}
{"type":"error","message":"..."}
```

`actual_remaining_time_s` is intentionally unavailable during a genuinely live run. It is a post-journey ground-truth label and is not fabricated by the dashboard.

## Validation

Run the existing simulator smoke test and the new backend tests:

```bash
python smoke_test.py
pytest backend/tests -q
```

Then validate the frontend:

```bash
cd frontend
npm run typecheck
npm run build
```

The backend regression tests include route-geometry serialization, clean reset behavior, rejection of arbitrary filesystem paths, API session creation, and the requirement that simulator telemetry is independent of UI playback rate.
