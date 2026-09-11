from __future__ import annotations

from dataclasses import dataclass, field

from .guardrails import weather_braking_factor, weather_speed_factor
from .models import CrossingState, SignalAspect, SimulationConfig, TelemetryFrame, TrainDirection, WeatherCondition
from .physics import kmh_to_ms, ms_to_kmh, safe_speed_for_target, update_velocity


@dataclass
class RuntimeTrain:
    train: object
    journey: object
    departure_time_s: float
    station_stops: list
    source_m: float
    destination_m: float
    route_position_m: float
    speed_kmh: float = 0.0
    acceleration_ms2: float = 0.0
    completed: bool = False
    dwell_until_s: float | None = None
    dwelling_station_id: str | None = None
    served_stations: set[str] = field(default_factory=set)

    @property
    def direction(self) -> TrainDirection:
        return TrainDirection.FORWARD if self.destination_m > self.source_m else TrainDirection.REVERSE

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == TrainDirection.FORWARD else -1.0


@dataclass
class Target:
    position_m: float
    speed_kmh: float
    reason: str
    hard_stop: bool = False
    station_id: str | None = None


class NetworkSimulationEngine:
    """Parallel-track multi-train simulation with occupancy-driven signalling."""

    def __init__(self, config: SimulationConfig, scenario_id: str = "default"):
        self.config = config
        self.scenario_id = scenario_id
        self.tick_count = 0
        self.sim_time_s = 0.0
        self.trains: list[RuntimeTrain] = []
        self._weather = {x.block_id: x for x in config.environment.weather}
        self._signal_schedules = {x.signal_id: x for x in config.environment.signal_states}

        self._add_train(config.train, config.journey, 0.0, config.primary_station_stops)
        for run in config.additional_train_runs:
            self._add_train(run.train, run.journey, run.departure_time_s, run.station_stops)

    def _endpoint_m(self, endpoint) -> float:
        return self.config.route.block_start_distance_m(endpoint.block_id) + endpoint.position_in_block_m

    def _add_train(self, train, journey, departure, stops) -> None:
        source = self._endpoint_m(journey.source)
        destination = self._endpoint_m(journey.destination)
        self.trains.append(RuntimeTrain(
            train=train,
            journey=journey,
            departure_time_s=departure,
            station_stops=list(stops),
            source_m=source,
            destination_m=destination,
            route_position_m=source,
            speed_kmh=train.initial_state.initial_speed_kmh,
        ))

    @property
    def is_complete(self) -> bool:
        return all(t.completed for t in self.trains)

    @staticmethod
    def _active_entry(timeline, sim_time_s: float):
        active = timeline[0]
        for item in timeline:
            if item.start_time_s <= sim_time_s:
                active = item
            else:
                break
        return active

    def _weather_at(self, block_id: str) -> tuple[WeatherCondition, float]:
        schedule = self._weather.get(block_id)
        if schedule is None:
            return WeatherCondition.CLEAR, 10000.0
        item = self._active_entry(schedule.timeline, self.sim_time_s)
        return item.condition, item.visibility_m

    def _crossing_state(self, crossing) -> CrossingState:
        return self._active_entry(crossing.timeline, self.sim_time_s).state

    def _restriction_active(self, restriction) -> bool:
        return self.sim_time_s >= restriction.start_time_s and (restriction.end_time_s is None or self.sim_time_s < restriction.end_time_s)

    def _signal_position(self, signal) -> float:
        start = self.config.route.block_start_distance_m(signal.protected_block_id)
        block = self.config.route.blocks[self.config.route.block_index(signal.protected_block_id)]
        return start if signal.direction == TrainDirection.FORWARD else start + block.length_m

    def _body_interval(self, train: RuntimeTrain) -> tuple[float, float]:
        if train.direction == TrainDirection.FORWARD:
            return train.route_position_m - train.train.length_m, train.route_position_m
        return train.route_position_m, train.route_position_m + train.train.length_m

    def _block_occupied(self, track_id: str, block_index: int, exclude: RuntimeTrain | None = None) -> bool:
        if block_index < 0 or block_index >= len(self.config.route.blocks):
            return False
        block = self.config.route.blocks[block_index]
        start = self.config.route.block_start_distance_m(block.block_id)
        end = start + block.length_m
        for train in self.trains:
            if train is exclude or train.train.track_id != track_id or train.completed or self.sim_time_s < train.departure_time_s:
                continue
            rear, front = self._body_interval(train)
            if front >= start - 1e-6 and rear <= end + 1e-6:
                return True
        return False

    def _scheduled_signal(self, signal_id: str) -> SignalAspect:
        schedule = self._signal_schedules.get(signal_id)
        if schedule is None:
            return SignalAspect.GREEN
        return self._active_entry(schedule.timeline, self.sim_time_s).aspect

    def signal_aspect(self, signal, exclude: RuntimeTrain | None = None) -> SignalAspect:
        if not self.config.dynamic_signalling:
            return self._scheduled_signal(signal.signal_id)
        idx = self.config.route.block_index(signal.protected_block_id)
        if self._block_occupied(signal.track_id, idx, exclude=exclude):
            return SignalAspect.RED
        next_idx = idx + (1 if signal.direction == TrainDirection.FORWARD else -1)
        if self._block_occupied(signal.track_id, next_idx, exclude=exclude):
            return SignalAspect.YELLOW
        return SignalAspect.GREEN

    def signal_states(self) -> dict[str, SignalAspect]:
        return {signal.signal_id: self.signal_aspect(signal) for signal in self.config.signals}

    def _ahead(self, train: RuntimeTrain, target_m: float) -> float | None:
        distance = (target_m - train.route_position_m) * train.sign
        return distance if distance >= -1e-6 else None

    def _platform_position(self, station_id: str, track_id: str) -> float | None:
        station = next((s for s in self.config.stations if s.station_id == station_id), None)
        if station is None:
            return None
        platform = next((p for p in station.platforms if p.track_id == track_id), None)
        if platform is None:
            return None
        return self.config.route.block_start_distance_m(platform.block_id) + platform.position_in_block_m

    def _current_ceiling(self, train: RuntimeTrain) -> tuple[float, WeatherCondition, float]:
        _, block, local = self.config.route.locate(train.route_position_m)
        weather, visibility = self._weather_at(block.block_id)
        ceiling = min(train.train.max_speed_kmh, block.speed_limit_kmh)
        if block.curve_speed_limit_kmh is not None:
            ceiling = min(ceiling, block.curve_speed_limit_kmh)
        ceiling *= weather_speed_factor(weather, visibility)
        for restriction in [*self.config.environment.temporary_speed_restrictions, *self.config.environment.maintenance_restrictions]:
            if restriction.block_id == block.block_id and restriction.start_position_m <= local < restriction.end_position_m and self._restriction_active(restriction):
                ceiling = min(ceiling, restriction.speed_limit_kmh)
        return max(0.0, ceiling), weather, visibility

    def _targets(self, train: RuntimeTrain) -> list[Target]:
        targets = [Target(train.destination_m, 0.0, "DESTINATION", True)]

        for stop in train.station_stops:
            if stop.station_id in train.served_stations:
                continue
            pos = self._platform_position(stop.station_id, train.train.track_id)
            if pos is not None and self._ahead(train, pos) is not None:
                targets.append(Target(pos, 0.0, f"STATION:{stop.station_id}", True, stop.station_id))

        for crossing in self.config.environment.crossings:
            if self._crossing_state(crossing) != CrossingState.CLOSED_FOR_TRAIN:
                continue
            pos = self.config.route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m
            if self._ahead(train, pos) is not None:
                targets.append(Target(pos, 0.0, f"CROSSING:{crossing.crossing_id}", True))

        for signal in self.config.signals:
            if signal.track_id != train.train.track_id or signal.direction != train.direction:
                continue
            pos = self._signal_position(signal)
            d = self._ahead(train, pos)
            if d is None:
                continue
            # Excluding this train prevents its own occupied block from making the
            # entry signal under its nose red. Other trains still make it red.
            if self.signal_aspect(signal, exclude=train) == SignalAspect.RED:
                targets.append(Target(pos, 0.0, f"RED_SIGNAL:{signal.signal_id}", True))

        separation = self.config.simulation.train_separation_m
        for other in self.trains:
            if other is train or other.completed or other.train.track_id != train.train.track_id or other.direction != train.direction:
                continue
            gap = (other.route_position_m - train.route_position_m) * train.sign
            if gap <= 0:
                continue
            safe_pos = other.route_position_m - train.sign * (other.train.length_m + separation)
            if self._ahead(train, safe_pos) is not None:
                targets.append(Target(safe_pos, 0.0, f"TRAIN_AHEAD:{other.train.train_id}", True))

        idx, current_block, _ = self.config.route.locate(train.route_position_m)
        future_indices = range(idx + 1, len(self.config.route.blocks)) if train.sign > 0 else range(idx - 1, -1, -1)
        for i in future_indices:
            block = self.config.route.blocks[i]
            start = self.config.route.block_start_distance_m(block.block_id)
            boundary = start if train.sign > 0 else start + block.length_m
            weather, visibility = self._weather_at(block.block_id)
            limit = min(train.train.max_speed_kmh, block.speed_limit_kmh)
            if block.curve_speed_limit_kmh is not None:
                limit = min(limit, block.curve_speed_limit_kmh)
            limit *= weather_speed_factor(weather, visibility)
            if limit < current_block.speed_limit_kmh or block.curve_speed_limit_kmh is not None:
                targets.append(Target(boundary, limit, f"UPCOMING_BLOCK:{block.block_id}"))
        return [t for t in targets if self._ahead(train, t.position_m) is not None]

    def _desired_speed(self, train: RuntimeTrain, ceiling: float, weather: WeatherCondition) -> tuple[float, str]:
        desired, reason = ceiling, "CURRENT_SPEED_CEILING"
        _, block, _ = self.config.route.locate(train.route_position_m)
        grade = 9.81 * block.gradient_percent / 100.0 * train.sign
        decel = max(0.2, train.train.service_decel_ms2 * weather_braking_factor(weather) + grade)
        margin = self.config.simulation.braking_safety_margin_m
        for target in self._targets(train):
            distance = max(0.0, (target.position_m - train.route_position_m) * train.sign - margin)
            safe = ms_to_kmh(safe_speed_for_target(kmh_to_ms(target.speed_kmh), distance, decel))
            if safe < desired:
                desired, reason = safe, target.reason
        return max(0.0, desired), reason

    def _advance(self, train: RuntimeTrain, dt: float) -> tuple[str, str]:
        if train.completed:
            return "STOP", "DESTINATION"
        if self.sim_time_s < train.departure_time_s:
            train.speed_kmh = 0.0
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
        new_pos = train.route_position_m + train.sign * travel

        hard = [t for t in self._targets(train) if t.hard_stop]
        if hard:
            nearest = min(hard, key=lambda t: (t.position_m - train.route_position_m) * train.sign)
            if (new_pos - nearest.position_m) * train.sign >= 0:
                new_pos = nearest.position_m
                new_v = 0.0
                action, reason = "STOP", nearest.reason
                if nearest.reason == "DESTINATION":
                    train.completed = True
                elif nearest.station_id:
                    stop = next(s for s in train.station_stops if s.station_id == nearest.station_id)
                    train.dwelling_station_id = nearest.station_id
                    train.dwell_until_s = self.sim_time_s + stop.dwell_time_s
                else:
                    new_pos -= train.sign * 0.01

        train.route_position_m = max(0.0, min(self.config.route.total_length_m, new_pos))
        train.speed_kmh = ms_to_kmh(new_v)
        train.acceleration_ms2 = 0.0 if train.completed else accel
        return action, reason

    def tick(self) -> list[TelemetryFrame]:
        if self.is_complete:
            return []
        dt = self.config.simulation.tick_seconds
        actions = {t.train.train_id: self._advance(t, dt) for t in self.trains}
        self.tick_count += 1
        self.sim_time_s += dt
        return [self.snapshot_train(t, *actions[t.train.train_id]) for t in self.trains]

    def snapshot_all(self) -> list[TelemetryFrame]:
        return [self.snapshot_train(t) for t in self.trains]

    def snapshot_train(self, train: RuntimeTrain, action: str = "INITIAL", reason: str = "INITIAL_STATE") -> TelemetryFrame:
        idx, block, local = self.config.route.locate(train.route_position_m)
        ceiling, weather, visibility = self._current_ceiling(train)
        distance = max(0.0, (train.destination_m - train.route_position_m) * train.sign)
        progress = min(1.0, abs(train.route_position_m - train.source_m) / max(1e-6, abs(train.destination_m - train.source_m)))

        next_signals = []
        for signal in self.config.signals:
            if signal.track_id != train.train.track_id or signal.direction != train.direction:
                continue
            pos = self._signal_position(signal)
            d = self._ahead(train, pos)
            if d is not None and d > 1e-6:
                next_signals.append((d, self.signal_aspect(signal, exclude=train)))
        next_signals.sort(key=lambda x: x[0])

        next_speed = next_speed_d = next_curve = next_curve_d = None
        future_indices = range(idx + 1, len(self.config.route.blocks)) if train.sign > 0 else range(idx - 1, -1, -1)
        for i in future_indices:
            future = self.config.route.blocks[i]
            start = self.config.route.block_start_distance_m(future.block_id)
            boundary = start if train.sign > 0 else start + future.length_m
            d = max(0.0, (boundary - train.route_position_m) * train.sign)
            if next_speed is None and future.speed_limit_kmh != block.speed_limit_kmh:
                next_speed, next_speed_d = future.speed_limit_kmh, d
            if next_curve is None and future.curve_speed_limit_kmh is not None:
                next_curve, next_curve_d = future.curve_speed_limit_kmh, d

        next_tsr = None
        for r in self.config.environment.temporary_speed_restrictions:
            if not self._restriction_active(r):
                continue
            start = self.config.route.block_start_distance_m(r.block_id)
            pos = start + (r.start_position_m if train.sign > 0 else r.end_position_m)
            d = self._ahead(train, pos)
            if d is not None and (next_tsr is None or d < next_tsr[0]):
                next_tsr = (d, r.speed_limit_kmh)

        next_crossing = None
        for c in self.config.environment.crossings:
            pos = self.config.route.block_start_distance_m(c.block_id) + c.position_in_block_m
            d = self._ahead(train, pos)
            if d is not None and (next_crossing is None or d < next_crossing[0]):
                next_crossing = (d, self._crossing_state(c))

        return TelemetryFrame(
            scenario_id=self.scenario_id,
            train_id=train.train.train_id,
            tick=self.tick_count,
            sim_time_s=self.sim_time_s,
            track_id=train.train.track_id,
            direction=train.direction,
            active=self.sim_time_s >= train.departure_time_s and not train.completed,
            completed=train.completed,
            current_station_id=train.dwelling_station_id,
            current_block_id=block.block_id,
            position_in_block_m=local,
            route_position_m=train.route_position_m,
            speed_kmh=train.speed_kmh,
            acceleration_ms2=train.acceleration_ms2,
            control_action=action,
            control_reason=reason,
            current_block_speed_limit_kmh=block.speed_limit_kmh,
            current_gradient_percent=block.gradient_percent,
            current_curve_speed_limit_kmh=block.curve_speed_limit_kmh,
            effective_speed_ceiling_kmh=ceiling,
            weather=weather,
            visibility_m=visibility,
            distance_to_destination_m=distance,
            route_progress=progress,
            next_signal_aspect=next_signals[0][1] if len(next_signals) > 0 else None,
            distance_to_next_signal_m=next_signals[0][0] if len(next_signals) > 0 else None,
            second_signal_aspect=next_signals[1][1] if len(next_signals) > 1 else None,
            distance_to_second_signal_m=next_signals[1][0] if len(next_signals) > 1 else None,
            next_speed_limit_kmh=next_speed,
            distance_to_next_speed_change_m=next_speed_d,
            next_curve_limit_kmh=next_curve,
            distance_to_next_curve_m=next_curve_d,
            next_tsr_limit_kmh=next_tsr[1] if next_tsr else None,
            distance_to_next_tsr_m=next_tsr[0] if next_tsr else None,
            next_crossing_state=next_crossing[1] if next_crossing else None,
            distance_to_next_crossing_m=next_crossing[0] if next_crossing else None,
        )
