"""
Batch scenario generator.

Runs the SimulationEngine many times over randomized anomaly conditions
(different signal holds, weather, speed restrictions, at different blocks
and different tick offsets) to produce a diverse training dataset.

This is deliberately config-driven (see ScenarioGeneratorConfig) rather than
hardcoded, so you can widen/narrow the randomization space without touching
engine code.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from .models import SimulationConfig, AnomalyType, SignalAspect, WeatherCondition
from .engine import SimulationEngine
from .anomaly import AnomalyInjector, AnomalyRequest
from .exporters import BatchExporter


@dataclass
class ScenarioGeneratorConfig:
    n_scenarios: int = 100
    anomaly_probability: float = 0.6         # chance a scenario includes an anomaly at all
    possible_anomaly_types: list[AnomalyType] = field(
        default_factory=lambda: list(AnomalyType)
    )
    min_anomaly_tick: int = 5
    max_anomaly_tick_fraction: float = 0.7    # anomaly injected within first N% of the run
    random_seed: int | None = None


class ScenarioGenerator:
    def __init__(self, base_config: SimulationConfig, gen_config: ScenarioGeneratorConfig):
        self.base_config = base_config
        self.gen_config = gen_config
        self._rng = random.Random(gen_config.random_seed)

    def _random_anomaly_request(self, config: SimulationConfig) -> AnomalyRequest:
        anomaly_type = self._rng.choice(self.gen_config.possible_anomaly_types)
        block = self._rng.choice(config.corridor.blocks)

        if anomaly_type == AnomalyType.SIGNAL_ASPECT_CHANGE:
            return AnomalyRequest(
                anomaly_type=anomaly_type,
                block_id=block.block_id,
                signal_aspect=self._rng.choice(
                    [SignalAspect.YELLOW, SignalAspect.RED, SignalAspect.DOUBLE_YELLOW]
                ),
            )
        if anomaly_type == AnomalyType.WEATHER_MODIFIER:
            return AnomalyRequest(
                anomaly_type=anomaly_type,
                block_id=block.block_id,
                weather=self._rng.choice([WeatherCondition.RAIN, WeatherCondition.HEAVY_FOG]),
            )
        if anomaly_type == AnomalyType.TEMPORARY_SPEED_RESTRICTION:
            return AnomalyRequest(
                anomaly_type=anomaly_type,
                block_id=block.block_id,
                speed_restriction_kmh=float(self._rng.choice([20, 30, 40, 50])),
            )
        # BLOCK_MAINTENANCE_HOLD
        return AnomalyRequest(
            anomaly_type=anomaly_type,
            block_id=block.block_id,
            maintenance_hold=True,
        )

    def run(self) -> BatchExporter:
        exporter = BatchExporter()

        for i in range(self.gen_config.n_scenarios):
            scenario_id = f"scenario_{i:05d}"
            config = copy.deepcopy(self.base_config)
            engine = SimulationEngine(config, scenario_id=scenario_id)
            injector = AnomalyInjector(config.corridor, config.caution_speed_kmh)

            inject_anomaly = self._rng.random() < self.gen_config.anomaly_probability
            anomaly_tick = None
            anomaly_request = None
            if inject_anomaly:
                max_tick = int(config.max_ticks * self.gen_config.max_anomaly_tick_fraction)
                anomaly_tick = self._rng.randint(self.gen_config.min_anomaly_tick, max(max_tick, self.gen_config.min_anomaly_tick + 1))
                anomaly_request = self._random_anomaly_request(config)

            for tick_num in range(config.max_ticks):
                if inject_anomaly and tick_num == anomaly_tick:
                    # inject against the first train in the config (extend as needed for multi-train scenarios)
                    train_id = config.trains[0].train_id
                    state = engine.train_states[train_id]
                    injector.inject(
                        anomaly_request,
                        train_speed_kmh=state.speed_kmh,
                        train_current_block_id=config.corridor.blocks[state.block_index].block_id,
                        emergency_decel_ms2=state.config.emergency_decel_ms2,
                    )

                frames = engine.tick()
                exporter.add(frames)

        return exporter
