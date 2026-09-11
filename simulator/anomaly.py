"""
Anomaly injection subsystem.

This is intentionally the SAME code path whether it's called:
  - programmatically, in a loop, with randomized parameters (batch training
    data generation), or
  - by a human clicking a button in a live control panel (demo mode)

That's the point: one function signature, two callers. Guardrails are
enforced here so neither caller can bypass them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import AnomalyType, SignalAspect, WeatherCondition, Corridor
from .guardrails import evaluate_signal_override, GuardrailResult


@dataclass
class AnomalyRequest:
    anomaly_type: AnomalyType
    block_id: str
    # payload — only the relevant field(s) need to be set depending on type
    signal_aspect: Optional[SignalAspect] = None
    weather: Optional[WeatherCondition] = None
    speed_restriction_kmh: Optional[float] = None
    maintenance_hold: Optional[bool] = None


@dataclass
class AnomalyResult:
    request: AnomalyRequest
    guardrail: GuardrailResult


class AnomalyInjector:
    """Thread-safe-by-design (no shared mutable state beyond the corridor
    object itself; caller is responsible for locking if calling concurrently
    from multiple request handlers later)."""

    def __init__(self, corridor: Corridor, caution_speed_kmh: float = 30.0):
        self.corridor = corridor
        self.caution_speed_kmh = caution_speed_kmh

    def inject(
        self,
        request: AnomalyRequest,
        train_speed_kmh: float,
        train_current_block_id: str,
        emergency_decel_ms2: float,
    ) -> AnomalyResult:
        block_idx = self.corridor.block_index(request.block_id)
        block = self.corridor.blocks[block_idx]

        # Guardrail only applies to signal-RED injections on the immediate
        # next block relative to the train's current position.
        if request.anomaly_type == AnomalyType.SIGNAL_ASPECT_CHANGE:
            current_idx = self.corridor.block_index(train_current_block_id)
            is_next_block = block_idx == current_idx + 1

            if is_next_block and request.signal_aspect == SignalAspect.RED:
                current_block = self.corridor.blocks[current_idx]
                remaining_m = current_block.length_m  # simplified: full block remaining
                result = evaluate_signal_override(
                    requested_aspect=request.signal_aspect,
                    train_speed_kmh=train_speed_kmh,
                    emergency_decel_ms2=emergency_decel_ms2,
                    remaining_block_distance_m=remaining_m,
                    next_block_id=self.corridor.blocks[block_idx + 1].block_id
                    if block_idx + 1 < len(self.corridor.blocks) else block.block_id,
                )
                if not result.allowed:
                    # Redirect: apply the hold to the further-out block instead
                    redirected = self.corridor.blocks[self.corridor.block_index(result.redirected_block_id)]
                    redirected.signal_aspect = SignalAspect.RED
                    return AnomalyResult(request=request, guardrail=result)

            block.signal_aspect = request.signal_aspect
            return AnomalyResult(request=request, guardrail=GuardrailResult(allowed=True))

        if request.anomaly_type == AnomalyType.WEATHER_MODIFIER:
            block.weather = request.weather
            return AnomalyResult(request=request, guardrail=GuardrailResult(allowed=True))

        if request.anomaly_type == AnomalyType.TEMPORARY_SPEED_RESTRICTION:
            block.temporary_speed_restriction_kmh = request.speed_restriction_kmh
            return AnomalyResult(request=request, guardrail=GuardrailResult(allowed=True))

        if request.anomaly_type == AnomalyType.BLOCK_MAINTENANCE_HOLD:
            block.maintenance_hold = bool(request.maintenance_hold)
            return AnomalyResult(request=request, guardrail=GuardrailResult(allowed=True))

        raise ValueError(f"Unhandled anomaly type: {request.anomaly_type}")
