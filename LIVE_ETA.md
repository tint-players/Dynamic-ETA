# Live hybrid ETA

The dashboard backend can now run the trained LSTM + GraphSAGE ETA model on the same inference-safe feature path used during offline training.

## Checkpoint

The selected checkpoint comes from GitHub Actions run `34871556440`, artifact `eta-experiment-20runs-12epochs-2s`, where the hybrid model won validation MAE and was evaluated once on the held-out test runs.

The backend looks for the checkpoint at:

```text
artifacts/eta_winner.pt
```

or at the path supplied by:

```bash
export DYNAMIC_ETA_ML_CHECKPOINT=/absolute/path/to/eta_winner.pt
```

To retrieve the currently selected artifact with GitHub CLI:

```bash
mkdir -p artifacts
gh run download 34871556440 \
  -n eta-experiment-20runs-12epochs-2s \
  -D artifacts
```

## Dependencies

PyTorch intentionally remains optional for the core simulator. For an ML-enabled dashboard install both backend and ML dependencies:

```bash
pip install -r backend/requirements.txt
pip install -r requirements-ml.txt
```

If PyTorch or the checkpoint is missing, the simulator still runs and the dashboard reports that live ML ETA is unavailable instead of failing startup.

## Runtime contract

For every live telemetry update the backend:

1. keeps the simulator's full one-second frame history,
2. uses `MLSampleBuilder.build_inputs()` for the target train,
3. therefore applies the same 60-second sequence, 16-node graph, topology, route mask and train-context construction used during training,
4. loads only the hybrid checkpoint's model weights and validated feature contract,
5. returns `remaining_time_s` and `arrival_simulation_s` for each active, incomplete train.

Post-run ground-truth fields are not read by live inference. The WebSocket `telemetry_batch` message and initial session response include an `eta_state` object. The frontend displays the selected train's live prediction above the existing dashboard.
