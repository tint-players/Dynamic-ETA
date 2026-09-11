from __future__ import annotations

import math


def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh / 3.6


def ms_to_kmh(v_ms: float) -> float:
    return v_ms * 3.6


def update_position(s_m: float, v_ms: float, a_ms2: float, dt_s: float) -> float:
    return s_m + (v_ms * dt_s) + (0.5 * a_ms2 * dt_s ** 2)


def update_velocity(v_ms: float, a_ms2: float, dt_s: float) -> float:
    return max(0.0, v_ms + a_ms2 * dt_s)


def braking_distance(v_from_ms: float, v_to_ms: float, decel_ms2: float) -> float:
    if decel_ms2 <= 0:
        raise ValueError("decel_ms2 must be positive")
    if v_from_ms <= v_to_ms:
        return 0.0
    return (v_from_ms ** 2 - v_to_ms ** 2) / (2 * decel_ms2)


def safe_speed_for_target(target_speed_ms: float, distance_m: float, decel_ms2: float) -> float:
    if decel_ms2 <= 0:
        raise ValueError("decel_ms2 must be positive")
    return math.sqrt(max(0.0, target_speed_ms ** 2 + 2 * decel_ms2 * max(0.0, distance_m)))


def emergency_braking_distance(v_ms: float, decel_ms2: float) -> float:
    return braking_distance(v_ms, 0.0, decel_ms2)
