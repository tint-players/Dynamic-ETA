from __future__ import annotations

from backend.session import SessionManager
from simulator.ml_contract import (
    ML_FORBIDDEN_LIVE_INPUT_FIELDS,
    attach_run_provenance,
    new_run_id,
    validate_labelled_training_frames,
)
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from simulator.scenario_generator import ScenarioGenerator, ScenarioGeneratorConfig
from simulator.dataset import label_completed_multi_train_journey
from simulator.track_blocks import track_block_id


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def test_run_ids_are_unique_and_opaque():
    first = new_run_id()
    second = new_run_id()

    assert first.startswith("run_")
    assert second.startswith("run_")
    assert first != second


def test_provenance_survives_post_run_labelling():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-contract-check")

    frames = engine.snapshot_all()
    while not engine.is_complete:
        frames.extend(engine.tick())

    frames = attach_run_provenance(frames, run_id="run_test", random_seed=123)
    labelled = validate_labelled_training_frames(label_completed_multi_train_journey(frames))

    assert labelled
    assert {frame.run_id for frame in labelled} == {"run_test"}
    assert {frame.random_seed for frame in labelled} == {123}
    assert all(frame.actual_remaining_time_s is not None for frame in labelled)
    assert all(frame.actual_remaining_time_s >= 0 for frame in labelled)


def test_post_run_labels_are_forbidden_live_inputs():
    assert ML_FORBIDDEN_LIVE_INPUT_FIELDS == {
        "actual_remaining_time_s",
        "actual_arrival_simulation_s",
        "total_journey_time_s",
    }


def test_scenario_generator_exports_run_seed_and_track_block_identity():
    generator = ScenarioGenerator(
        _config(),
        ScenarioGeneratorConfig(
            n_scenarios=1,
            random_seed=4242,
            baseline_probability=1.0,
        ),
    )

    dataframe = generator.run().to_dataframe()

    assert not dataframe.empty
    assert dataframe["run_id"].notna().all()
    assert dataframe["run_id"].nunique() == 1
    assert set(dataframe["random_seed"].unique()) == {4242}
    assert dataframe["actual_remaining_time_s"].notna().all()
    assert all(
        row.track_block_id == track_block_id(row.track_id, row.current_block_id)
        for row in dataframe.itertuples()
    )
