"""Operational modifiers used by Component A.

Anomaly-injection guardrails are intentionally deferred to Component B.
"""
from __future__ import annotations

from .models import WeatherCondition

_BASE_SPEED_FACTOR = {
    WeatherCondition.CLEAR: 1.00,
    WeatherCondition.RAIN: 0.90,
    WeatherCondition.HEAVY_RAIN: 0.75,
    WeatherCondition.FOG: 0.80,
    WeatherCondition.HEAVY_FOG: 0.60,
}

_BRAKING_FACTOR = {
    WeatherCondition.CLEAR: 1.00,
    WeatherCondition.RAIN: 0.90,
    WeatherCondition.HEAVY_RAIN: 0.80,
    WeatherCondition.FOG: 1.00,
    WeatherCondition.HEAVY_FOG: 1.00,
}


def weather_speed_factor(condition: WeatherCondition, visibility_m: float) -> float:
    visibility_factor = 1.0
    if visibility_m < 200:
        visibility_factor = 0.50
    elif visibility_m < 500:
        visibility_factor = 0.60
    elif visibility_m < 1000:
        visibility_factor = 0.75
    elif visibility_m < 2000:
        visibility_factor = 0.85
    return min(_BASE_SPEED_FACTOR[condition], visibility_factor)


def weather_braking_factor(condition: WeatherCondition) -> float:
    return _BRAKING_FACTOR[condition]
