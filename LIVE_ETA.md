# Live hybrid ETA

The dashboard backend can run the trained LSTM + GraphSAGE ETA model on the same inference-safe feature path used during offline training.

## Selected checkpoint

The selected checkpoint comes from GitHub Actions run `34871556440`, artifact `eta-experiment-20runs-12epochs-2s`, where the hybrid model won validation MAE and was evaluated once on held-out test runs.

Expected checkpoint SHA256:

```text
24feb1ca5cc30d84698a331324588e4305dcb4915dee77dc4680a11507b18655
```

The default runtime path is:

```text
artifacts/eta_winner.pt
```

An alternate path can be supplied with:

```bash
export DYNAMIC_ETA_ML_CHECKPOINT=/absolute/path/to/eta_winner.pt
```

## Recommended setup

Install the backend and ML dependencies, then run the checkpoint installer:

```bash
pip install -r backend/requirements.txt
pip install -r requirements-ml.txt
python scripts/setup_live_eta.py
```

The setup script downloads the exact selected experiment artifact with GitHub CLI, places `eta_winner.pt` under `artifacts/`, and verifies its SHA256 before reporting success. It is idempotent: an already-installed valid checkpoint is simply verified.

The download path requires an installed/authenticated GitHub CLI (`gh`). If you already have the checkpoint locally, no GitHub download is needed:

```bash
python scripts/setup_live_eta.py --source /path/to/eta_winner.pt
```

Use `--force` only when intentionally replacing the destination file.

## Manual retrieval

The equivalent GitHub CLI command is:

```bash
mkdir -p artifacts
gh run download 34871556440 \
  -R tint-players/Dynamic-ETA \
  -n eta-experiment-20runs-12epochs-2s \
  -D artifacts
```

Then verify:

```bash
python scripts/setup_live_eta.py
```

## Runtime behavior

PyTorch intentionally remains optional for the core simulator. If PyTorch or the checkpoint is missing or invalid, the simulator still starts and the dashboard reports live ML ETA as unavailable instead of breaking simulation startup.

For every live telemetry update the backend:

1. keeps the simulator's full one-second frame history,
2. uses `MLSampleBuilder.build_inputs()` for the target train,
3. applies the same 60-second sequence, 16-node graph, topology, route mask and train-context construction used during training,
4. validates the checkpoint feature contract before loading weights,
5. returns `remaining_time_s` and `arrival_simulation_s` for each active, incomplete train.

Post-run ground-truth fields are not read by live inference. The WebSocket `telemetry_batch` message and initial session response include an `eta_state` object, and the frontend displays the selected train's live prediction above the existing dashboard.
