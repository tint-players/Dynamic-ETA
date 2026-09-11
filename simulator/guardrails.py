"""
Safety guardrails.

Two responsibilities:
  1. effective_speed_ceiling() — combine track limit, TSR, signal aspect,
     and weather into a single "you may not exceed this speed right now"
     number.
  2. evaluate_signal_override() — the EBD lockout check: refuse a RED signal
     injection on the immediate-next block if the train physically cannot
     stop in time, and push the hold one block further out instead.

Kept separate from the tick engine so the rules can be unit-tested and
tuned independently of the simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import SignalAspect, WeatherCondition, TrackBlock
from .physics import emergency_braking_distance

WEATHER_SPEED_CAP_KMH = {
    WeatherCondition.CLEAR: None,      # no cap
    WeatherCondition.RAIN: 80.0,
    WeatherCondition.HEAVY_FOG: 60.0,
}


def aspect_speed_cap_kmh(aspect: SignalAspect, caution_speed_kmh: float) -> Optional[float]:
    if aspect == SignalAspect.RED:
        return 0.0
    if aspect in (SignalAspect.YELLOW, SignalAspect.DOUBLE_YELLOW):
        return caution_speed_kmh
    return None  # GREEN -> no cap from signal


def effective_speed_ceiling(
    block: TrackBlock,
    caution_speed_kmh: float,
) -> float:
    """min(track limit, TSR, signal aspect cap, weather cap)."""
    candidates = [block.max_speed_kmh]

    if block.temporary_speed_restriction_kmh is not None:
        candidates.append(block.temporary_speed_restriction_kmh)

    aspect_cap = aspect_speed_cap_kmh(block.signal_aspect, caution_speed_kmh)
    if aspect_cap is not None:
        candidates.append(aspect_cap)

    weather_cap = WEATHER_SPEED_CAP_KMH.get(block.weather)
    if weather_cap is not None:
        candidates.append(weather_cap)

    return min(candidates)


@dataclass
class GuardrailResult:
    allowed: bool
    event: Optional[str] = None
    redirected_block_id: Optional[str] = None


def evaluate_signal_override(
    requested_aspect: SignalAspect,
    train_speed_kmh: float,
    emergency_decel_ms2: float,
    remaining_block_distance_m: float,
    next_block_id: str,
) -> GuardrailResult:
    """
    FR-2.3: If a RED aspect is requested on the immediate next block (Block +1)
    and the train cannot physically stop within the remaining distance to that
    block boundary, reject the override, log it, and signal that the hold
    should be shifted one block further out (Block +2).
    """
    if requested_aspect != SignalAspect.RED:
        return GuardrailResult(allowed=True)

    v_ms = train_speed_kmh / 3.6
    d_ebd = emergency_braking_distance(v_ms, emergency_decel_ms2)

    if d_ebd >= remaining_block_distance_m:
        return GuardrailResult(
            allowed=False,
            event="PHYSICS_OVERRULE_LOCKOUT",
            redirected_block_id=next_block_id,
        )

    return GuardrailResult(allowed=True)
