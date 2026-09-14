from __future__ import annotations

from dataclasses import replace

import pytest

from backend.session import SessionManager
from simulator.ml_dataset import MLTrainingDatasetBuilder, split_training_samples_by_run
from simulator.ml_samples import MLSampleBuilder
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def _short_labelled_run(ticks: int = 3, run_id: str = "run-dataset-001"):
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-dataset")
    frames = engine.snapshot_all()
    for _ in range(ticks):
        frames.extend(engine.tick())

    labelled = [
        frame.model_copy(update={
            "run_id": run_id,
            "random_seed": 42,
            "actual_remaining_time_s": max(0.0, 1000.0 - frame.sim_time_s),
            "actual_arrival_simulation_s": 1000.0,
            "total_journey_time_s": 1000.0,
        })
        for frame in frames
    ]
    return config, labelled


def test_dataset_builder_emits_one_sample_per_active_train_per_second():
    config, frames = _short_labelled_run(ticks=3)
    builder = MLTrainingDatasetBuilder(config)

    samples = builder.build_run_samples(frames)

    expected = sum(1 for frame in frames if frame.active and not frame.completed)
    assert len(samples) == expected
    assert {sample.run_id for sample in samples} == {"run-dataset-001"}
    assert all(sample.y >= 0.0 for sample in samples)
    assert all(len(sample.inputs.x_seq) == 60 for sample in samples)


def test_dataset_builder_does_not_emit_predeparture_inactive_trains():
    config, frames = _short_labelled_run(ticks=2)
    builder = MLTrainingDatasetBuilder(config)

    samples = builder.build_run_samples(frames)

    # The DOWN train departs at t=55 and follow-up at t=150, so neither should
    # produce samples in this short initial run window.
    sampled_train_ids = {sample.train_id for sample in samples}
    assert "TRAIN-DOWN-01" not in sampled_train_ids
    assert "TRAIN-FOLLOW-UP" not in sampled_train_ids
    assert config.train.train_id in sampled_train_ids


def test_dataset_builder_rejects_multiple_run_ids():
    config, frames = _short_labelled_run(ticks=1)
    mixed = list(frames)
    mixed[-1] = mixed[-1].model_copy(update={"run_id": "run-other"})

    with pytest.raises(ValueError, match="exactly one run_id"):
        MLTrainingDatasetBuilder(config).build_run_samples(mixed)


def _base_sample():
    config, frames = _short_labelled_run(ticks=1, run_id="run-base")
    current_time = max(frame.sim_time_s for frame in frames)
    current = [frame for frame in frames if abs(frame.sim_time_s - current_time) <= 1e-9]
    return MLSampleBuilder(config).build_training_sample(
        history=frames,
        current_frames=current,
        target_train_id=config.train.train_id,
    )


def test_run_split_never_leaks_one_run_across_partitions():
    base = _base_sample()
    samples = []
    for run_number in range(10):
        run_id = f"run-{run_number:02d}"
        for second in range(3):
            samples.append(replace(base, run_id=run_id, sim_time_s=float(second)))

    split = split_training_samples_by_run(samples, split_seed=123)

    train_ids = set(split.train_run_ids)
    validation_ids = set(split.validation_run_ids)
    test_ids = set(split.test_run_ids)
    assert not (train_ids & validation_ids)
    assert not (train_ids & test_ids)
    assert not (validation_ids & test_ids)
    assert train_ids | validation_ids | test_ids == {f"run-{i:02d}" for i in range(10)}
    assert len(split.train_run_ids) == 7
    assert len(split.validation_run_ids) + len(split.test_run_ids) == 3
    assert split.sample_count == len(samples)
    assert {sample.run_id for sample in split.train} == train_ids
    assert {sample.run_id for sample in split.validation} == validation_ids
    assert {sample.run_id for sample in split.test} == test_ids


def test_run_split_is_deterministic_for_seed_and_keeps_small_three_run_sets_nonempty():
    base = _base_sample()
    samples = [replace(base, run_id=f"run-{i}") for i in range(3)]

    first = split_training_samples_by_run(samples, split_seed=99)
    second = split_training_samples_by_run(reversed(samples), split_seed=99)

    assert first.train_run_ids == second.train_run_ids
    assert first.validation_run_ids == second.validation_run_ids
    assert first.test_run_ids == second.test_run_ids
    assert len(first.train_run_ids) == 1
    assert len(first.validation_run_ids) == 1
    assert len(first.test_run_ids) == 1
