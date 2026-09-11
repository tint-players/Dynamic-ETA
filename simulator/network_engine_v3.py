from __future__ import annotations

from .guardrails import weather_braking_factor
from .network_engine import NetworkSimulationEngine, RuntimeTrain
from .physics import kmh_to_ms, ms_to_kmh, safe_speed_for_target


class NetworkSimulationEngineV3(NetworkSimulationEngine):
    """Network engine with exact hard-stop approach behavior.

    Hard stops (stations, signals, crossings, destination) brake to the target
    itself. Applying the general braking safety margin to a zero-speed hard
    target can make a train settle several metres before it and never reach the
    stop. Non-zero future speed targets still use the configured margin.
    """

    def _desired_speed(self, train: RuntimeTrain, ceiling: float, weather):
        desired, reason = ceiling, "CURRENT_SPEED_CEILING"
        _, block, _ = self.config.route.locate(train.route_position_m)
        grade = 9.81 * block.gradient_percent / 100.0 * train.sign
        decel = max(0.2, train.train.service_decel_ms2 * weather_braking_factor(weather) + grade)
        margin = self.config.simulation.braking_safety_margin_m
        for target in self._targets(train):
            raw_distance = max(0.0, (target.position_m - train.route_position_m) * train.sign)
            distance = raw_distance if target.hard_stop else max(0.0, raw_distance - margin)
            safe = ms_to_kmh(safe_speed_for_target(kmh_to_ms(target.speed_kmh), distance, decel))
            if safe < desired:
                desired, reason = safe, target.reason
        return max(0.0, desired), reason
