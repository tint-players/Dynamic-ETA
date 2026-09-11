"""
Pure kinematic physics functions. No state, no I/O — just math, so it's
trivially testable and reusable regardless of whether it's driven by the
batch generator or a live tick loop.
"""

from __future__ import annotations


def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh / 3.6


def ms_to_kmh(v_ms: float) -> float:
    return v_ms * 3.6


def update_position(s_m: float, v_ms: float, a_ms2: float, dt_s: float) -> float:
    """Euler position update: S(t+dt) = S(t) + v*dt + 0.5*a*dt^2"""
    return s_m + (v_ms * dt_s) + (0.5 * a_ms2 * dt_s ** 2)


def update_velocity(v_ms: float, a_ms2: float, dt_s: float) -> float:
    """Euler velocity update, clamped at 0 (no reversing)."""
    return max(0.0, v_ms + a_ms2 * dt_s)


def emergency_braking_distance(v_ms: float, decel_ms2: float) -> float:
    """d_EBD = v^2 / (2 * decel). decel must be > 0."""
    if decel_ms2 <= 0:
        raise ValueError("decel_ms2 must be positive")
    return (v_ms ** 2) / (2 * decel_ms2)
