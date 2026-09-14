from __future__ import annotations

import pytest

from backend.session import SessionManager
from simulator.ml_contract import attach_run_provenance, validate_labelled_training_frames
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from simulator.dataset import label_completed_multi_train_journey


def _completed_frames():
    config = SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="ml-contract-validation")
    frames = engine.snapshot_all()
    while not engine.is_complete:
        frames.extend(engine.tick())
    return frames


def test_training_validation_rejects_unlabelled_frames():
    frames = attach_run_provenance(_completed_frames()[:1], run_id="run_test", random_seed=7)

    with pytest.raises(ValueError, match="actual_remaining_time_s"):
        validate_labelled_training_frames(frames)


def test_training_validation_rejects_missing_run_identity():
    labelled = label_completed_multi_train_journey(_completed_frames())

    with pytest.raises(ValueError, match="missing run_id"):
        validate_labelled_training_frames(labelled)
