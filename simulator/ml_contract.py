from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from .models import TelemetryFrame


ML_PROVENANCE_FIELDS = (
    "run_id",
    "scenario_id",
    "random_seed",
    "train_id",
    "sim_time_s",
    "track_id",
    "current_block_id",
)

ML_POST_RUN_LABEL_FIELDS = (
    "actual_remaining_time_s",
    "actual_arrival_simulation_s",
    "total_journey_time_s",
)

# These values are valid post-run labels/metrics, but must never be admitted to
# live model inputs or inference-time feature tensors.
ML_FORBIDDEN_LIVE_INPUT_FIELDS = frozenset(ML_POST_RUN_LABEL_FIELDS)


def new_run_id() -> str:
    """Return a globally unique opaque identifier for one simulation run."""
    return f"run_{uuid4().hex}"


def attach_run_provenance(
    frames: Iterable[TelemetryFrame],
    *,
    run_id: str,
    random_seed: int,
) -> list[TelemetryFrame]:
    """Attach immutable run-level provenance to raw telemetry frames.

    Engines are allowed to emit frames without ML provenance because live UI
    sessions and compatibility tests also consume TelemetryFrame. Any frame that
    enters the ML dataset path must pass through this function first.
    """
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")

    return [
        frame.model_copy(update={"run_id": run_id, "random_seed": random_seed})
        for frame in frames
    ]


def validate_labelled_training_frames(frames: Iterable[TelemetryFrame]) -> list[TelemetryFrame]:
    """Validate the minimum provenance and label contract for ML rows."""
    materialized = list(frames)
    for frame in materialized:
        if not frame.run_id:
            raise ValueError("ML training frame is missing run_id")
        if frame.random_seed is None:
            raise ValueError(f"ML training frame {frame.run_id} is missing random_seed")
        if frame.actual_remaining_time_s is None:
            raise ValueError(f"ML training frame {frame.run_id} is missing actual_remaining_time_s")
        if frame.actual_arrival_simulation_s is None:
            raise ValueError(f"ML training frame {frame.run_id} is missing actual_arrival_simulation_s")
        if frame.total_journey_time_s is None:
            raise ValueError(f"ML training frame {frame.run_id} is missing total_journey_time_s")
        if frame.actual_remaining_time_s < 0:
            raise ValueError(f"ML training frame {frame.run_id} has negative remaining time")
    return materialized
