from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Iterable, Protocol

from .ml_features import SEQUENCE_FEATURE_NAMES, SPEED_SCALE_KMH
from .ml_samples import MLModelInputs, MLTrainingSample
from .models import SimulationConfig


_SPEED_INDEX = SEQUENCE_FEATURE_NAMES.index("speed_norm")
_DISTANCE_INDEX = SEQUENCE_FEATURE_NAMES.index("distance_to_destination_ratio")
_HISTORY_PRESENT_INDEX = SEQUENCE_FEATURE_NAMES.index("history_present")
_EFFECTIVE_CEILING_INDEX = SEQUENCE_FEATURE_NAMES.index("effective_speed_ceiling_norm")


class ETAPredictor(Protocol):
    """Minimal prediction interface shared by baselines and future neural models."""

    def predict_seconds(self, inputs: MLModelInputs) -> float:
        ...


@dataclass(frozen=True, slots=True)
class CurrentSpeedETABaseline:
    """Naive ETA from remaining distance and current speed.

    A configurable speed floor keeps the prediction finite when the train is
    temporarily stopped. The model uses only inference-safe inputs from the
    current sequence row.
    """

    route_length_m: float
    speed_floor_kmh: float = 10.0

    @classmethod
    def from_config(cls, config: SimulationConfig, *, speed_floor_kmh: float = 10.0) -> "CurrentSpeedETABaseline":
        return cls(route_length_m=float(config.route.total_length_m), speed_floor_kmh=speed_floor_kmh)

    def predict_seconds(self, inputs: MLModelInputs) -> float:
        latest = inputs.x_seq[-1]
        remaining_m = max(0.0, latest[_DISTANCE_INDEX] * self.route_length_m)
        speed_kmh = max(self.speed_floor_kmh, latest[_SPEED_INDEX] * SPEED_SCALE_KMH)
        return remaining_m / (speed_kmh / 3.6)


@dataclass(frozen=True, slots=True)
class HistoricalMeanSpeedETABaseline:
    """Naive ETA using the mean observed speed in the available history window."""

    route_length_m: float
    speed_floor_kmh: float = 10.0

    @classmethod
    def from_config(cls, config: SimulationConfig, *, speed_floor_kmh: float = 10.0) -> "HistoricalMeanSpeedETABaseline":
        return cls(route_length_m=float(config.route.total_length_m), speed_floor_kmh=speed_floor_kmh)

    def predict_seconds(self, inputs: MLModelInputs) -> float:
        latest = inputs.x_seq[-1]
        remaining_m = max(0.0, latest[_DISTANCE_INDEX] * self.route_length_m)
        observed_speeds = [
            row[_SPEED_INDEX] * SPEED_SCALE_KMH
            for row in inputs.x_seq
            if row[_HISTORY_PRESENT_INDEX] > 0.5
        ]
        mean_speed_kmh = (
            sum(observed_speeds) / len(observed_speeds)
            if observed_speeds
            else self.speed_floor_kmh
        )
        speed_kmh = max(self.speed_floor_kmh, mean_speed_kmh)
        return remaining_m / (speed_kmh / 3.6)


@dataclass(frozen=True, slots=True)
class EffectiveCeilingETABaseline:
    """Optimistic ETA assuming the current effective speed ceiling is sustained."""

    route_length_m: float
    speed_floor_kmh: float = 10.0

    @classmethod
    def from_config(cls, config: SimulationConfig, *, speed_floor_kmh: float = 10.0) -> "EffectiveCeilingETABaseline":
        return cls(route_length_m=float(config.route.total_length_m), speed_floor_kmh=speed_floor_kmh)

    def predict_seconds(self, inputs: MLModelInputs) -> float:
        latest = inputs.x_seq[-1]
        remaining_m = max(0.0, latest[_DISTANCE_INDEX] * self.route_length_m)
        ceiling_kmh = max(self.speed_floor_kmh, latest[_EFFECTIVE_CEILING_INDEX] * SPEED_SCALE_KMH)
        return remaining_m / (ceiling_kmh / 3.6)


@dataclass(frozen=True, slots=True)
class ETAErrorBucket:
    name: str
    sample_count: int
    mae_s: float
    median_absolute_error_s: float
    p90_absolute_error_s: float


@dataclass(frozen=True, slots=True)
class ETAMetrics:
    sample_count: int
    mae_s: float
    median_absolute_error_s: float
    p90_absolute_error_s: float
    buckets: tuple[ETAErrorBucket, ...]


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _error_summary(name: str, errors: list[float]) -> ETAErrorBucket:
    if not errors:
        return ETAErrorBucket(name, 0, 0.0, 0.0, 0.0)
    return ETAErrorBucket(
        name=name,
        sample_count=len(errors),
        mae_s=sum(errors) / len(errors),
        median_absolute_error_s=float(median(errors)),
        p90_absolute_error_s=_percentile(errors, 0.90),
    )


def evaluate_eta_predictor(
    predictor: ETAPredictor,
    samples: Iterable[MLTrainingSample],
) -> ETAMetrics:
    """Evaluate a predictor with global and remaining-ETA bucketed errors.

    Buckets use ground-truth remaining time only for evaluation, never as a
    model input: 0-5 min, 5-15 min, 15-30 min, and 30+ min.
    """

    materialized = list(samples)
    errors: list[float] = []
    bucket_errors: dict[str, list[float]] = {
        "0-5min": [],
        "5-15min": [],
        "15-30min": [],
        "30min+": [],
    }

    for sample in materialized:
        prediction = float(predictor.predict_seconds(sample.inputs))
        if not isfinite(prediction) or prediction < 0:
            raise ValueError("ETA predictor must return a finite non-negative number of seconds")
        error = abs(prediction - sample.y)
        errors.append(error)
        if sample.y < 5 * 60:
            bucket_errors["0-5min"].append(error)
        elif sample.y < 15 * 60:
            bucket_errors["5-15min"].append(error)
        elif sample.y < 30 * 60:
            bucket_errors["15-30min"].append(error)
        else:
            bucket_errors["30min+"].append(error)

    if not errors:
        return ETAMetrics(
            sample_count=0,
            mae_s=0.0,
            median_absolute_error_s=0.0,
            p90_absolute_error_s=0.0,
            buckets=tuple(_error_summary(name, []) for name in bucket_errors),
        )

    return ETAMetrics(
        sample_count=len(errors),
        mae_s=sum(errors) / len(errors),
        median_absolute_error_s=float(median(errors)),
        p90_absolute_error_s=_percentile(errors, 0.90),
        buckets=tuple(_error_summary(name, bucket_errors[name]) for name in bucket_errors),
    )
