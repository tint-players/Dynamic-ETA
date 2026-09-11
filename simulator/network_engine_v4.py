from __future__ import annotations

from .guardrails import weather_braking_factor
from .models import CrossingState
from .network_engine import RuntimeTrain, Target
from .network_engine_v3 import NetworkSimulationEngineV3
from .physics import kmh_to_ms, ms_to_kmh, update_velocity


class NetworkSimulationEngineV4(NetworkSimulationEngineV3):
    """Network engine with crossover turnarounds and train-triggered level crossings.

    Level crossings are controlled from the road-user perspective:
    - road open => CLOSED_FOR_TRAIN
    - road closed/protected => OPEN_FOR_TRAIN

    A train entering the approach zone starts a gate-closing delay. Until the
    road is fully closed the train receives a hard stop target before the road,
    never on the crossing itself. The road remains closed until the entire train
    plus a clearance margin has passed, followed by a short reopening delay.
    """

    CROSSING_APPROACH_TRIGGER_M = 600.0
    CROSSING_STOP_LINE_M = 60.0
    CROSSING_GATE_CLOSING_S = 12.0
    CROSSING_CLEARANCE_MARGIN_M = 20.0
    CROSSING_REOPEN_DELAY_S = 5.0

    def __init__(self, config, scenario_id: str = "default"):
        super().__init__(config, scenario_id=scenario_id)
        self._crossing_runtime = {
            crossing.crossing_id: {
                "phase": "ROAD_OPEN",
                "close_ready_at": None,
                "reopen_ready_at": None,
            }
            for crossing in self.config.environment.crossings
        }
        self._update_crossings()

    def _crossing_position(self, crossing) -> float:
        return self.config.route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m

    def _crossing_has_demand(self, crossing) -> bool:
        crossing_pos = self._crossing_position(crossing)
        for train in self.trains:
            if train.completed or self.sim_time_s < train.departure_time_s:
                continue
            signed_front_distance = (crossing_pos - train.route_position_m) * train.sign
            lower_bound = -(train.train.length_m + self.CROSSING_CLEARANCE_MARGIN_M)
            if lower_bound <= signed_front_distance <= self.CROSSING_APPROACH_TRIGGER_M:
                return True
        return False

    def _update_crossings(self) -> None:
        for crossing in self.config.environment.crossings:
            runtime = self._crossing_runtime[crossing.crossing_id]
            demand = self._crossing_has_demand(crossing)
            phase = runtime["phase"]

            if phase == "ROAD_OPEN":
                if demand:
                    runtime["phase"] = "CLOSING"
                    runtime["close_ready_at"] = self.sim_time_s + self.CROSSING_GATE_CLOSING_S
            elif phase == "CLOSING":
                if not demand:
                    runtime["phase"] = "ROAD_OPEN"
                    runtime["close_ready_at"] = None
                elif self.sim_time_s >= float(runtime["close_ready_at"]):
                    runtime["phase"] = "PROTECTED"
                    runtime["close_ready_at"] = None
            elif phase == "PROTECTED":
                if not demand:
                    runtime["phase"] = "CLEARING"
                    runtime["reopen_ready_at"] = self.sim_time_s + self.CROSSING_REOPEN_DELAY_S
            elif phase == "CLEARING":
                if demand:
                    runtime["phase"] = "PROTECTED"
                    runtime["reopen_ready_at"] = None
                elif self.sim_time_s >= float(runtime["reopen_ready_at"]):
                    runtime["phase"] = "ROAD_OPEN"
                    runtime["reopen_ready_at"] = None

    def _crossing_state(self, crossing) -> CrossingState:
        runtime = self._crossing_runtime.get(crossing.crossing_id)
        if runtime is None:
            return super()._crossing_state(crossing)
        return CrossingState.OPEN_FOR_TRAIN if runtime["phase"] in {"PROTECTED", "CLEARING"} else CrossingState.CLOSED_FOR_TRAIN

    def crossing_states(self) -> dict[str, CrossingState]:
        return {crossing.crossing_id: self._crossing_state(crossing) for crossing in self.config.environment.crossings}

    def crossing_phases(self) -> dict[str, str]:
        return {crossing_id: str(runtime["phase"]) for crossing_id, runtime in self._crossing_runtime.items()}

    def _targets(self, train: RuntimeTrain) -> list[Target]:
        targets = super()._targets(train)
        adjusted: list[Target] = []
        for target in targets:
            if not target.reason.startswith("CROSSING:"):
                adjusted.append(target)
                continue
            crossing_id = target.reason.split(":", 1)[1]
            crossing = next(item for item in self.config.environment.crossings if item.crossing_id == crossing_id)
            crossing_pos = self._crossing_position(crossing)
            stop_line = crossing_pos - train.sign * self.CROSSING_STOP_LINE_M
            if self._ahead(train, stop_line) is None and self._ahead(train, crossing_pos) is not None:
                stop_line = train.route_position_m
            adjusted.append(Target(stop_line, 0.0, target.reason, True))
        return [target for target in adjusted if self._ahead(train, target.position_m) is not None]

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

    def tick(self):
        if self.is_complete:
            return []
        self._update_crossings()
        dt = self.config.simulation.tick_seconds
        actions = {train.train.train_id: self._advance(train, dt) for train in self.trains}
        self.tick_count += 1
        self.sim_time_s += dt
        self._update_crossings()
        return [self.snapshot_train(train, *actions[train.train.train_id]) for train in self.trains]
