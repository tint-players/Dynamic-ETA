from __future__ import annotations

from .guardrails import weather_braking_factor
from .network_engine import RuntimeTrain
from .network_engine_v3 import NetworkSimulationEngineV3
from .physics import kmh_to_ms, ms_to_kmh, update_velocity


class NetworkSimulationEngineV4(NetworkSimulationEngineV3):
    """Network engine with physical crossover turnarounds.

    A track-change plan may request ``reverse_after_change``. The train then
    crosses at the configured crossover midpoint, stops, switches to the target
    track, and reverses toward its original source. Non-turnaround crossovers
    retain the v3 behavior and continue in the same travel direction.
    """

    def _apply_track_change_v4(self, train: RuntimeTrain, old_pos: float, new_pos: float) -> tuple[str | None, bool]:
        for plan in train.track_changes:
            if plan.crossover_id in train.completed_crossovers:
                continue
            crossover = self._crossovers[plan.crossover_id]
            if train.current_track_id != crossover.from_track_id:
                continue
            _, _, midpoint = self._crossover_bounds(crossover)
            crossed = (old_pos - midpoint) * train.sign < 0 <= (new_pos - midpoint) * train.sign
            if not crossed:
                continue

            train.current_track_id = crossover.to_track_id
            train.completed_crossovers.add(plan.crossover_id)
            if plan.reverse_after_change:
                original_source = train.source_m
                train.route_position_m = midpoint
                train.source_m = midpoint
                train.destination_m = original_source
                return f"TURNAROUND:{plan.crossover_id}", True
            return f"CROSSOVER:{plan.crossover_id}", False
        return None, False

    def _advance(self, train: RuntimeTrain, dt: float) -> tuple[str, str]:
        if train.completed:
            return "STOP", "DESTINATION"
        if self.sim_time_s < train.departure_time_s:
            train.speed_kmh = 0.0
            train.acceleration_ms2 = 0.0
            return "WAIT", "SCHEDULED_DEPARTURE"
        if train.dwell_until_s is not None:
            if self.sim_time_s < train.dwell_until_s:
                train.speed_kmh = 0.0
                train.acceleration_ms2 = 0.0
                return "STOP", f"STATION_DWELL:{train.dwelling_station_id}"
            if train.dwelling_station_id:
                train.served_stations.add(train.dwelling_station_id)
            train.dwelling_station_id = None
            train.dwell_until_s = None

        ceiling, weather, _ = self._current_ceiling(train)
        desired, reason = self._desired_speed(train, ceiling, weather)
        _, block, _ = self.config.route.locate(train.route_position_m)
        grade = 9.81 * block.gradient_percent / 100.0 * train.sign
        tol = self.config.simulation.speed_tolerance_kmh
        if train.speed_kmh > desired + tol:
            accel, action = -max(0.2, train.train.service_decel_ms2 * weather_braking_factor(weather) + grade), "BRAKE"
        elif train.speed_kmh < desired - tol:
            accel, action = max(0.05, train.train.accel_ms2 - grade), "ACCELERATE"
        else:
            accel, action = 0.0, "MAINTAIN"

        old_v = kmh_to_ms(train.speed_kmh)
        new_v = update_velocity(old_v, accel, dt)
        desired_v = kmh_to_ms(desired)
        new_v = min(new_v, desired_v) if accel > 0 else max(new_v, desired_v) if accel < 0 else new_v
        travel = max(0.0, old_v * dt + 0.5 * accel * dt * dt)
        old_pos = train.route_position_m
        new_pos = old_pos + train.sign * travel

        hard = [target for target in self._targets(train) if target.hard_stop]
        if hard:
            nearest = min(hard, key=lambda target: (target.position_m - old_pos) * train.sign)
            if (new_pos - nearest.position_m) * train.sign >= 0:
                new_pos = nearest.position_m
                new_v = 0.0
                action, reason = "STOP", nearest.reason
                if nearest.reason == "DESTINATION":
                    train.completed = True
                elif nearest.station_id:
                    stop = next(item for item in train.station_stops if item.station_id == nearest.station_id)
                    train.dwelling_station_id = nearest.station_id
                    train.dwell_until_s = self.sim_time_s + stop.dwell_time_s
                else:
                    new_pos -= train.sign * 0.01

        train.route_position_m = max(0.0, min(self.config.route.total_length_m, new_pos))
        crossover_reason, turned_around = self._apply_track_change_v4(train, old_pos, train.route_position_m)
        if turned_around:
            new_v = 0.0
            accel = 0.0
            action = "STOP"
            reason = crossover_reason or "TURNAROUND"
        elif crossover_reason and action != "STOP":
            reason = crossover_reason

        train.speed_kmh = ms_to_kmh(new_v)
        train.acceleration_ms2 = 0.0 if train.completed or turned_around else accel
        return action, reason
