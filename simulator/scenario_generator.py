from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from .dataset import label_completed_journey
from .engine import SimulationEngine
from .exporters import BatchExporter
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


@dataclass
class ScenarioGeneratorConfig:
    n_scenarios: int = 100
    random_seed: int | None = None
    baseline_probability: float = 0.30
    tsr_probability: float = 0.20
    crossing_closure_probability: float = 0.30
    signal_restriction_probability: float = 0.20


class ScenarioGenerator:
    """Generate normal/controlled operating scenarios for Component A."""

    def __init__(self, base_config: SimulationConfig, gen_config: ScenarioGeneratorConfig):
        self.base_config = base_config
        self.gen_config = gen_config
        self._rng = random.Random(gen_config.random_seed)

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
        block = self._rng.choice(config.route.blocks)
        start = block.length_m * self._rng.uniform(0.15, 0.55)
        end = min(block.length_m, start + block.length_m * self._rng.uniform(0.15, 0.35))
        limit = min(block.speed_limit_kmh * self._rng.uniform(0.45, 0.75), block.speed_limit_kmh - 5)
        config.environment.temporary_speed_restrictions = [
            TemporarySpeedRestriction(
                restriction_id="GEN-TSR-001",
                block_id=block.block_id,
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
            return config
        self._random_weather(config)
        self._random_tsr(config)
        self._random_crossings(config)
        self._random_signals(config)
        return config

    def run(self) -> BatchExporter:
        exporter = BatchExporter()
        for i in range(self.gen_config.n_scenarios):
            scenario_id = f"scenario_{i:05d}"
            config = self._make_scenario()
            frames = SimulationEngine(config, scenario_id=scenario_id).run()
            exporter.add(label_completed_journey(frames))
        return exporter
