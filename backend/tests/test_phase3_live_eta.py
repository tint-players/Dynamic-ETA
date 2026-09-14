from __future__ import annotations

from pathlib import Path

import pytest

from backend.live_eta import live_eta_state, reset_runtime_cache_for_tests
from backend.session import sessions


def test_live_eta_degrades_cleanly_when_checkpoint_is_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DYNAMIC_ETA_ML_CHECKPOINT", str(tmp_path / "missing.pt"))
    reset_runtime_cache_for_tests()
    session = sessions.create("delhi_agra_corridor.yaml")

    state = live_eta_state(session, session.snapshots())

    assert state["available"] is False
    assert state["predictions"] == {}
    assert "checkpoint not found" in state["message"]


def test_trained_hybrid_checkpoint_produces_live_inference(monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.delenv("DYNAMIC_ETA_ML_CHECKPOINT", raising=False)
    reset_runtime_cache_for_tests()
    session = sessions.create("delhi_agra_corridor.yaml")

    state = live_eta_state(session, session.snapshots())
    if not state["predictions"]:
        frames = session.tick()
        state = live_eta_state(session, frames)

    assert state["available"] is True
    assert state["model_name"] == "hybrid"
    assert state["predictions"]
    for prediction in state["predictions"].values():
        assert prediction["remaining_time_s"] >= 0.0
        assert prediction["arrival_simulation_s"] >= prediction["remaining_time_s"]
