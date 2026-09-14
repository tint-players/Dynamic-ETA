from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence

from .ml_contract import validate_labelled_training_frames
from .ml_samples import MLSampleBuilder, MLTrainingSample
from .models import SimulationConfig, TelemetryFrame


@dataclass(frozen=True, slots=True)
class MLDatasetSplit:
    """Leakage-safe train/validation/test partition grouped by simulation run."""

    train: tuple[MLTrainingSample, ...]
    validation: tuple[MLTrainingSample, ...]
    test: tuple[MLTrainingSample, ...]
    train_run_ids: tuple[str, ...]
    validation_run_ids: tuple[str, ...]
    test_run_ids: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


class MLTrainingDatasetBuilder:
    """Build one labelled ETA sample per active train per simulation second.

    The input telemetry must already represent a completed and post-run-labelled
    simulation run. Samples are emitted only while a train is active and before
    it is marked completed. This keeps the training cadence aligned with live
    inference: one prediction opportunity for every observable active second.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.sample_builder = MLSampleBuilder(config)

    def build_run_samples(self, frames: Iterable[TelemetryFrame]) -> tuple[MLTrainingSample, ...]:
        labelled = validate_labelled_training_frames(frames)
        if not labelled:
            return ()

        run_ids = {frame.run_id for frame in labelled}
        if len(run_ids) != 1:
            raise ValueError("build_run_samples requires telemetry from exactly one run_id")

        # A run must not silently mix scenario identities or randomization seeds.
        scenario_ids = {frame.scenario_id for frame in labelled}
        random_seeds = {frame.random_seed for frame in labelled}
        if len(scenario_ids) != 1:
            raise ValueError("One run_id contains multiple scenario_id values")
        if len(random_seeds) != 1:
            raise ValueError("One run_id contains multiple random_seed values")

        ordered = sorted(labelled, key=lambda frame: (frame.sim_time_s, frame.tick, frame.train_id))
        by_step: dict[tuple[int, float], list[TelemetryFrame]] = defaultdict(list)
        for frame in ordered:
            by_step[(frame.tick, frame.sim_time_s)].append(frame)

        history: list[TelemetryFrame] = []
        samples: list[MLTrainingSample] = []
        for step_key in sorted(by_step, key=lambda item: (item[1], item[0])):
            current = sorted(by_step[step_key], key=lambda frame: frame.train_id)
            history.extend(current)
            for target in current:
                if not target.active or target.completed:
                    continue
                samples.append(
                    self.sample_builder.build_training_sample(
                        history=history,
                        current_frames=current,
                        target_train_id=target.train_id,
                    )
                )

        return tuple(samples)


def _split_counts(total_runs: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    if total_runs < 0:
        raise ValueError("total_runs must be non-negative")
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, validation, and test values")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios cannot be negative")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")

    normalized = [ratio / total_ratio for ratio in ratios]
    positive = [index for index, ratio in enumerate(normalized) if ratio > 0]

    counts = [0, 0, 0]
    remaining = total_runs
    if total_runs >= len(positive):
        for index in positive:
            counts[index] = 1
        remaining -= len(positive)

    if remaining:
        raw = [remaining * ratio for ratio in normalized]
        floors = [int(value) for value in raw]
        for index, value in enumerate(floors):
            counts[index] += value
        leftovers = remaining - sum(floors)
        priority = sorted(
            range(3),
            key=lambda index: (raw[index] - floors[index], normalized[index], -index),
            reverse=True,
        )
        for index in priority[:leftovers]:
            counts[index] += 1

    return counts[0], counts[1], counts[2]


def split_training_samples_by_run(
    samples: Iterable[MLTrainingSample],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
) -> MLDatasetSplit:
    """Partition samples without ever splitting a run across datasets."""

    materialized = list(samples)
    grouped: dict[str, list[MLTrainingSample]] = defaultdict(list)
    for sample in materialized:
        if not sample.run_id:
            raise ValueError("ML training sample is missing run_id")
        grouped[sample.run_id].append(sample)

    run_ids = sorted(grouped)
    Random(split_seed).shuffle(run_ids)
    n_train, n_validation, n_test = _split_counts(
        len(run_ids),
        (train_ratio, validation_ratio, test_ratio),
    )

    train_ids = tuple(run_ids[:n_train])
    validation_ids = tuple(run_ids[n_train:n_train + n_validation])
    test_ids = tuple(run_ids[n_train + n_validation:n_train + n_validation + n_test])

    def collect(ids: tuple[str, ...]) -> tuple[MLTrainingSample, ...]:
        return tuple(
            sample
            for run_id in ids
            for sample in sorted(grouped[run_id], key=lambda item: (item.sim_time_s, item.train_id))
        )

    return MLDatasetSplit(
        train=collect(train_ids),
        validation=collect(validation_ids),
        test=collect(test_ids),
        train_run_ids=train_ids,
        validation_run_ids=validation_ids,
        test_run_ids=test_ids,
    )
