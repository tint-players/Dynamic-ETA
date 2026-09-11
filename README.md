# Railway Telemetry Simulator — Phase 1

A standalone, dependency-light simulator core. No web framework, no ML —
just physics, safety guardrails, and two output modes. Built this way so a
FastAPI/WebSocket layer and the ETA model can be plugged in later without
touching this code.

## Structure

```
simulator/
  models.py             # Pydantic data models (Track, Train, Telemetry, enums)
  physics.py             # Pure Euler-integration kinematics + EBD formula
  guardrails.py           # Effective speed ceiling + EBD signal-override lockout
  anomaly.py             # Anomaly injection (shared by batch & live callers)
  engine.py              # The tick loop — advances trains one step at a time
  exporters.py            # BatchExporter (Parquet/CSV) + LiveExporter (JSON)
  scenario_generator.py    # Randomized batch runs -> training dataset
  config_loader.py         # YAML -> validated SimulationConfig
examples/
  delhi_agra_corridor.yaml # Example corridor + train definition
output/                    # Generated datasets land here
smoke_test.py             # End-to-end demonstration script
```

## Why it's split this way

- **`physics.py`** has zero dependencies on anything else — pure math,
  easy to unit test, easy to swap (e.g. for a fancier integration method later).
- **`guardrails.py`** and **`anomaly.py`** are separate from `engine.py` so
  safety rules can change without touching the tick loop, and vice versa.
- **`engine.py`** doesn't know or care whether it's called in a batch for-loop
  or from a future async API route — it just exposes `.tick()`.
- **`exporters.py`** is the only place that knows about JSON/Parquet — this
  answers the "is JSON mandatory?" question: it isn't, and it's isolated
  to one file so you can add e.g. a Kafka/Arrow exporter later in one place.
- **Config is YAML** validated through Pydantic (`models.py`), so you (or a
  teammate) can add a new corridor or train by copying `examples/*.yaml`
  and editing values — no code changes needed. The same models also accept
  plain Python dicts, so a future API request body works identically.

## Two ways to run it

**Batch (generate training data):**
```python
from simulator import load_simulation_config, ScenarioGenerator, ScenarioGeneratorConfig

config = load_simulation_config("examples/delhi_agra_corridor.yaml")
generator = ScenarioGenerator(config, ScenarioGeneratorConfig(n_scenarios=500))
exporter = generator.run()
exporter.to_parquet("output/training_data.parquet")
```

**Live (drive it manually / from a future API):**
```python
from simulator import load_simulation_config, SimulationEngine, LiveExporter

config = load_simulation_config("examples/delhi_agra_corridor.yaml")
engine = SimulationEngine(config, scenario_id="demo")
frames = engine.tick()
json_str = LiveExporter.to_json(frames[0])  # ready to push over WebSocket
```

## Not yet built (by design — comes after this)
- FastAPI routes / WebSocket server
- Redis Pub/Sub bridge
- Neo4j graph sync
- The actual ETA prediction model
- Frontend control panel

## Run the smoke test
```
pip install pydantic pyyaml pandas pyarrow
python3 smoke_test.py
```
