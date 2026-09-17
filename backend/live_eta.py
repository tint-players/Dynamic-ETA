from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "artifacts" / "eta_winner.pt"


class LiveETARuntime:
    """Lazy, inference-only wrapper around the trained hybrid ETA checkpoint.

    PyTorch remains an optional dependency for the simulator/dashboard. Importing
    this module never imports torch; loading the checkpoint does. That preserves
    the existing lightweight backend path while allowing ML-enabled deployments
    to install ``requirements.txt`` and get live ETA predictions.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        import torch

        from simulator.ml_features import (
            CONTEXT_FEATURE_NAMES,
            GRAPH_FEATURE_NAMES,
            SEQUENCE_FEATURE_NAMES,
            SEQUENCE_WINDOW_STEPS,
        )
        from simulator.ml_hybrid import HybridLSTMGraphSAGEETARegressor, TorchHybridETAPredictor

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("ETA checkpoint payload must be a dictionary")
        if payload.get("format_version") != 1:
            raise ValueError(f"Unsupported ETA checkpoint format: {payload.get('format_version')!r}")
        if payload.get("model_name") != "hybrid":
            raise ValueError(f"Live ETA requires a hybrid checkpoint, got {payload.get('model_name')!r}")

        contract = payload.get("feature_contract") or {}
        expected_contract = {
            "sequence_window_steps": SEQUENCE_WINDOW_STEPS,
            "sequence_feature_names": tuple(SEQUENCE_FEATURE_NAMES),
            "graph_feature_names": tuple(GRAPH_FEATURE_NAMES),
            "context_feature_names": tuple(CONTEXT_FEATURE_NAMES),
        }
        for key, expected in expected_contract.items():
            actual = contract.get(key)
            if key.endswith("_feature_names") and actual is not None:
                actual = tuple(actual)
            if actual != expected:
                raise ValueError(f"ETA checkpoint feature contract mismatch for {key}")

        model_kwargs = dict(payload.get("model_kwargs") or {})
        model = HybridLSTMGraphSAGEETARegressor(**model_kwargs)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()

        self.checkpoint_path = checkpoint_path
        self.model_name = str(payload["model_name"])
        self.predictor = TorchHybridETAPredictor(model, device="cpu")
        self._builders: dict[int, Any] = {}

    def _builder(self, config):
        from simulator.ml_samples import MLSampleBuilder

        key = id(config)
        builder = self._builders.get(key)
        if builder is None:
            builder = MLSampleBuilder(config)
            self._builders = {key: builder}
        return builder

    def predict(self, *, config, history, current_frames, target_train_id: str) -> float:
        builder = self._builder(config)
        inputs = builder.build_inputs(
            history=history,
            current_frames=current_frames,
            target_train_id=target_train_id,
        )
        return self.predictor.predict_seconds(inputs)


_runtime_lock = Lock()
_runtime: LiveETARuntime | None = None
_runtime_error: str | None = None
_runtime_path: Path | None = None


def configured_checkpoint_path() -> Path:
    raw = os.environ.get("DYNAMIC_ETA_ML_CHECKPOINT")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_CHECKPOINT_PATH


def _get_runtime() -> tuple[LiveETARuntime | None, str | None]:
    global _runtime, _runtime_error, _runtime_path

    checkpoint_path = configured_checkpoint_path()
    with _runtime_lock:
        if _runtime_path == checkpoint_path and (_runtime is not None or _runtime_error is not None):
            return _runtime, _runtime_error

        _runtime = None
        _runtime_error = None
        _runtime_path = checkpoint_path
        if not checkpoint_path.exists():
            _runtime_error = f"checkpoint not found: {checkpoint_path}"
            return None, _runtime_error
        try:
            _runtime = LiveETARuntime(checkpoint_path)
        except Exception as exc:  # optional torch or invalid checkpoint should not break the simulator
            _runtime_error = str(exc)
        return _runtime, _runtime_error


def live_eta_state(session, current_frames) -> dict[str, Any]:
    """Return live ETA predictions for the current simulator state.

    Predictions are built from the exact same ``MLSampleBuilder.build_inputs``
    path used by offline training. Ground-truth post-run labels are never read.
    """

    frames = list(current_frames)
    runtime, error = _get_runtime()
    if runtime is None:
        return {
            "available": False,
            "model_name": None,
            "message": error or "live ETA unavailable",
            "predictions": {},
        }

    predictions: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for frame in frames:
        if not frame.active or frame.completed:
            continue
        try:
            remaining_s = max(0.0, runtime.predict(
                config=session.config,
                history=session.frames,
                current_frames=frames,
                target_train_id=frame.train_id,
            ))
            predictions[frame.train_id] = {
                "remaining_time_s": remaining_s,
                "arrival_simulation_s": float(frame.sim_time_s + remaining_s),
            }
        except Exception as exc:
            failures.append(f"{frame.train_id}: {exc}")

    message = None if not failures else "; ".join(failures)
    return {
        "available": True,
        "model_name": runtime.model_name,
        "message": message,
        "predictions": predictions,
    }


def reset_runtime_cache_for_tests() -> None:
    global _runtime, _runtime_error, _runtime_path
    with _runtime_lock:
        _runtime = None
        _runtime_error = None
        _runtime_path = None
