from __future__ import annotations

from backend.session import SessionManager
from simulator.network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from simulator.scenario_generator import ScenarioGenerator, ScenarioGeneratorConfig
from simulator.track_aware_exporters import TrackAwareBatchExporter
from simulator.track_blocks import get_track_block, track_block_id


def _config():
    return SessionManager.load_scenario("delhi_agra_corridor.yaml").model_copy(deep=True)


def test_random_tsr_is_generated_on_one_canonical_track_block():
    config = _config()
    generator = ScenarioGenerator(
        config,
        ScenarioGeneratorConfig(
            n_scenarios=1,
            random_seed=7,
            baseline_probability=0.0,
            tsr_probability=1.0,
            crossing_closure_probability=0.0,
            signal_restriction_probability=0.0,
        ),
    )

    generated = generator._make_scenario()
    assert len(generated.environment.temporary_speed_restrictions) == 1

    restriction = generated.environment.temporary_speed_restrictions[0]
    assert restriction.track_id in generated.route.track_ids
    section = get_track_block(generated.route, restriction.track_id, restriction.block_id)
    assert section.track_id == restriction.track_id
    assert section.block_id == restriction.block_id


def test_delhi_agra_generator_uses_restrictive_network_engine():
    config = _config()
    generator = ScenarioGenerator(config, ScenarioGeneratorConfig(n_scenarios=1, random_seed=11))

    engine = generator._new_engine(config, "generator-engine-check")

    assert isinstance(engine, NetworkSimulationEngineV4Restrictive)
    assert len(engine.trains) == 4


def test_track_aware_batch_exporter_adds_canonical_identity():
    config = _config()
    engine = NetworkSimulationEngineV4Restrictive(config, scenario_id="batch-track-block-check")
    exporter = TrackAwareBatchExporter()
    exporter.add(engine.snapshot_all())
    dataframe = exporter.to_dataframe()

    assert "track_block_id" in dataframe.columns
    assert all(
        row.track_block_id == track_block_id(row.track_id, row.current_block_id)
        for row in dataframe.itertuples()
    )
