from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from .dataset import label_completed_journey, label_completed_multi_train_journey
from .engine import SimulationEngine
from .ml_contract import attach_run_provenance, new_run_id, validate_labelled_training_frames
from .models import (
    BlockWeatherSchedule,
    CrossingState,
    CrossingTimelineEntry,
    SignalAspect,
    SignalStateSchedule,
    SignalTimelineEntry,
    SimulationConfig,
    TemporarySpeedRestriction,
    WeatherCondition,
    WeatherTimelineEntry,
)
from .network_engine_v4_restrictive import NetworkSimulationEngineV4Restrictive
from .track_aware_exporters import TrackAwareBatchExporter
from .track_blocks import enumerate_track_blocks
from .track_block_validation import validate_track_block_config


@dataclass
class ScenarioGeneratorConfig:
    n_scenarios: int = 100
    random_seed: int | None = None
    baseline_probability: float = 0.30
    tsr_probability: float = 0.20
    crossing_closure_probability: float = 0.30
    signal_restriction_probability: float = 0.20


class ScenarioGenerator:
    """Generate normal/controlled operating scenarios for Component A.

    Shared geographic effects such as weather remain logical-block scoped.
    Operational restrictions are generated against canonical track-block
    identities so one physical track can be restricted while the parallel track
    remains unaffected.
    """

    def __init__(self, base_config: SimulationConfig, gen_config: ScenarioGeneratorConfig):
        self.base_config = base_config
        self.gen_config = gen_config
        self.random_seed = (
            gen_config.random_seed
            if gen_config.random_seed is not None
            else random.SystemRandom().randrange(0, 2**63)
        )
        self._rng = random.Random(self.random_seed)

    @staticmethod
    def _requires_network_engine(config: SimulationConfig) -> bool:
        return (
            len(config.route.track_ids) > 1
            or bool(config.additional_train_runs)
            or bool(config.stations)
            or config.dynamic_signalling
        )

    @classmethod
    def _new_engine(cls, config: SimulationConfig, scenario_id: str):
        if cls._requires_network_engine(config):
            return NetworkSimulationEngineV4Restrictive(config, scenario_id=scenario_id)
        return SimulationEngine(config, scenario_id=scenario_id)

    @classmethod
    def _run_config(cls, config: SimulationConfig, scenario_id: str):
        engine = cls._new_engine(config, scenario_id)
        if isinstance(engine, SimulationEngine):
            return engine.run(), False

        frames = engine.snapshot_all()
        while not engine.is_complete and engine.sim_time_s < config.simulation.max_simulation_time_s:
            frames.extend(engine.tick())
        if not engine.is_complete:
            raise RuntimeError(
                f"Scenario {scenario_id} timed out at {engine.sim_time_s:.1f}s before all trains completed"
            )
        return frames, True

    def _random_weather(self, config: SimulationConfig) -> None:
        choices = [
            (WeatherCondition.CLEAR, 10000.0),
            (WeatherCondition.RAIN, 4000.0),
            (WeatherCondition.HEAVY_RAIN, 1800.0),
            (WeatherCondition.FOG, 700.0),
            (WeatherCondition.HEAVY_FOG, 250.0),
        ]
        weights = [0.55, 0.20, 0.05, 0.15, 0.05]
        config.environment.weather = [
            BlockWeatherSchedule(
                block_id=block.block_id,
                timeline=[WeatherTimelineEntry(start_time_s=0, condition=condition, visibility_m=visibility)],
            )
            for block in config.route.blocks
            for condition, visibility in [self._rng.choices(choices, weights=weights, k=1)[0]]
        ]

    def _baseline(self, config: SimulationConfig) -> None:
        config.environment.weather = [
            BlockWeatherSchedule(
                block_id=b.block_id,
                timeline=[WeatherTimelineEntry(start_time_s=0, condition=WeatherCondition.CLEAR, visibility_m=10000)],
            )
            for b in config.route.blocks
        ]
        config.environment.signal_states = [
            SignalStateSchedule(
                signal_id=s.signal_id,
                timeline=[SignalTimelineEntry(start_time_s=0, aspect=SignalAspect.GREEN)],
            )
            for s in config.signals
        ]
        config.environment.temporary_speed_restrictions = []
        config.environment.maintenance_restrictions = []
        config.environment.crossings = [
            c.model_copy(update={
                "timeline": [CrossingTimelineEntry(start_time_s=0, state=CrossingState.OPEN_FOR_TRAIN)]
            })
            for c in config.environment.crossings
        ]

    def _random_tsr(self, config: SimulationConfig) -> None:
        if self._rng.random() >= self.gen_config.tsr_probability:
            config.environment.temporary_speed_restrictions = []
            return

        section = self._rng.choice(enumerate_track_blocks(config.route))
        block = section.geometry
        start = block.length_m * self._rng.uniform(0.15, 0.55)
        end = min(block.length_m, start + block.length_m * self._rng.uniform(0.15, 0.35))
        limit = min(block.speed_limit_kmh * self._rng.uniform(0.45, 0.75), block.speed_limit_kmh - 5)
        config.environment.temporary_speed_restrictions = [
            TemporarySpeedRestriction(
                restriction_id="GEN-TSR-001",
                block_id=section.block_id,
                track_id=section.track_id,
                start_position_m=start,
                end_position_m=end,
                speed_limit_kmh=max(20.0, limit),
                start_time_s=0,
            )
        ]

    def _random_crossings(self, config: SimulationConfig) -> None:
        updated = []
        for crossing in config.environment.crossings:
            if self._rng.random() < self.gen_config.crossing_closure_probability:
                close_at = self._rng.uniform(30, 240)
                reopen_at = close_at + self._rng.uniform(20, 90)
                timeline = [
                    CrossingTimelineEntry(start_time_s=0, state=CrossingState.OPEN_FOR_TRAIN),
                    CrossingTimelineEntry(start_time_s=close_at, state=CrossingState.CLOSED_FOR_TRAIN),
                    CrossingTimelineEntry(start_time_s=reopen_at, state=CrossingState.OPEN_FOR_TRAIN),
                ]
            else:
                timeline = [CrossingTimelineEntry(start_time_s=0, state=CrossingState.OPEN_FOR_TRAIN)]
            updated.append(crossing.model_copy(update={"timeline": timeline}))
        config.environment.crossings = updated

    def _random_signals(self, config: SimulationConfig) -> None:
        schedules = {
            s.signal_id: [SignalTimelineEntry(start_time_s=0, aspect=SignalAspect.GREEN)]
            for s in config.signals
        }
        if len(config.signals) > 1 and self._rng.random() < self.gen_config.signal_restriction_probability:
            target_idx = self._rng.randint(1, len(config.signals) - 1)
            target = config.signals[target_idx]
            previous = config.signals[target_idx - 1]
            change_at = self._rng.uniform(40, 180)
            clear_at = change_at + self._rng.uniform(30, 100)
            schedules[target.signal_id] = [
                SignalTimelineEntry(start_time_s=0, aspect=SignalAspect.GREEN),
                SignalTimelineEntry(start_time_s=change_at, aspect=SignalAspect.RED),
                SignalTimelineEntry(start_time_s=clear_at, aspect=SignalAspect.GREEN),
            ]
            schedules[previous.signal_id] = [
                SignalTimelineEntry(start_time_s=0, aspect=SignalAspect.GREEN),
                SignalTimelineEntry(start_time_s=change_at, aspect=SignalAspect.YELLOW),
                SignalTimelineEntry(start_time_s=clear_at, aspect=SignalAspect.GREEN),
            ]
        config.environment.signal_states = [
            SignalStateSchedule(signal_id=s.signal_id, timeline=schedules[s.signal_id])
            for s in config.signals
        ]

    def _make_scenario(self) -> SimulationConfig:
        config = copy.deepcopy(self.base_config)
        if self._rng.random() < self.gen_config.baseline_probability:
            self._baseline(config)
            return validate_track_block_config(config)

        self._random_weather(config)
        self._random_tsr(config)
        self._random_crossings(config)
        self._random_signals(config)
        return validate_track_block_config(config)

    def run(self) -> TrackAwareBatchExporter:
        exporter = TrackAwareBatchExporter()
        for i in range(self.gen_config.n_scenarios):
            scenario_id = f"scenario_{i:05d}"
            run_id = new_run_id()
            config = self._make_scenario()
            frames, is_network = self._run_config(config, scenario_id)
            frames = attach_run_provenance(
                frames,
                run_id=run_id,
                random_seed=self.random_seed,
            )
            labelled = (
                label_completed_multi_train_journey(frames)
                if is_network
                else label_completed_journey(frames)
            )
            exporter.add(validate_labelled_training_frames(labelled))
        return exporter
