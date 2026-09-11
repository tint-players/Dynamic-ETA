"""
The tick engine — the thing that actually runs the simulation forward
in time, one tick per train per step.

This module has NO idea whether it's being driven by a batch scenario
loop or a live async server loop. Someone calls `.tick()` repeatedly;
that's the entire contract. That's what keeps this "versatile and easy
to integrate" — a future FastAPI route can call `.tick()` on a timer,
and the batch generator can call `.tick()` in a tight for-loop. Same engine.
"""

from __future__ import annotations

from typing import Optional

from .models import (
    SimulationConfig,
    TrainConfig,
    TelemetryFrame,
)
from .physics import kmh_to_ms, ms_to_kmh, update_position, update_velocity, emergency_braking_distance
from .guardrails import effective_speed_ceiling


class TrainState:
    """Mutable runtime state for one train (position within its current block, speed, etc.)."""

    def __init__(self, config: TrainConfig, block_index: int):
        self.config = config
        self.block_index = block_index
        self.position_in_block_m = 0.0
        self.speed_kmh = config.initial_speed_kmh
        self.acceleration_ms2 = config.accel_ms2


class SimulationEngine:
    def __init__(self, config: SimulationConfig, scenario_id: str = "default"):
        self.config = config
        self.scenario_id = scenario_id
        self.tick_count = 0

        self.train_states: dict[str, TrainState] = {}
        for train_cfg in config.trains:
            idx = config.corridor.block_index(train_cfg.start_block_id)
            self.train_states[train_cfg.train_id] = TrainState(train_cfg, idx)

    def tick(self) -> list[TelemetryFrame]:
        """Advance the simulation by one tick_seconds step for every train.
        Returns one TelemetryFrame per train."""
        frames: list[TelemetryFrame] = []
        dt = self.config.tick_seconds

        for train_id, state in self.train_states.items():
            block = self.config.corridor.blocks[state.block_index]

            ceiling_kmh = effective_speed_ceiling(block, self.config.caution_speed_kmh)
            ceiling_ms = kmh_to_ms(ceiling_kmh)

            v_ms = kmh_to_ms(state.speed_kmh)

            # Simple governor: accelerate toward ceiling, decelerate if over it.
            if v_ms < ceiling_ms:
                a_ms2 = state.config.accel_ms2
            elif v_ms > ceiling_ms:
                a_ms2 = -state.config.service_decel_ms2
            else:
                a_ms2 = 0.0

            new_v_ms = update_velocity(v_ms, a_ms2, dt)
            new_pos = update_position(state.position_in_block_m, v_ms, a_ms2, dt)

            guardrail_triggered = False
            guardrail_event: Optional[str] = None

            # Cross block boundary if we've gone past this block's length
            if new_pos >= block.length_m:
                overflow = new_pos - block.length_m
                if state.block_index + 1 < len(self.config.corridor.blocks):
                    state.block_index += 1
                    new_pos = overflow
                    block = self.config.corridor.blocks[state.block_index]
                else:
                    # end of corridor — clamp
                    new_pos = block.length_m
                    new_v_ms = 0.0

            state.position_in_block_m = new_pos
            state.speed_kmh = ms_to_kmh(new_v_ms)
            state.acceleration_ms2 = a_ms2

            ebd_m = emergency_braking_distance(new_v_ms, state.config.emergency_decel_ms2) if new_v_ms > 0 else 0.0
            remaining_block_m = block.length_m - state.position_in_block_m

            frames.append(
                TelemetryFrame(
                    scenario_id=self.scenario_id,
                    train_id=train_id,
                    tick=self.tick_count,
                    sim_time_s=self.tick_count * dt,
                    block_id=block.block_id,
                    position_in_block_m=state.position_in_block_m,
                    speed_kmh=state.speed_kmh,
                    acceleration_ms2=state.acceleration_ms2,
                    effective_speed_ceiling_kmh=ceiling_kmh,
                    signal_aspect=block.signal_aspect,
                    weather=block.weather,
                    ebd_distance_m=ebd_m,
                    remaining_block_distance_m=remaining_block_m,
                    anomaly_active=(
                        block.temporary_speed_restriction_kmh is not None
                        or block.maintenance_hold
                        or block.weather.value != "CLEAR"
                        or block.signal_aspect.value != "GREEN"
                    ),
                    guardrail_triggered=guardrail_triggered,
                    guardrail_event=guardrail_event,
                )
            )

        self.tick_count += 1
        return frames

    def run(self, max_ticks: Optional[int] = None) -> list[TelemetryFrame]:
        """Run the full simulation and collect all frames (batch use)."""
        n = max_ticks or self.config.max_ticks
        all_frames: list[TelemetryFrame] = []
        for _ in range(n):
            all_frames.extend(self.tick())
        return all_frames
