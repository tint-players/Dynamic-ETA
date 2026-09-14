from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .guardrails import weather_braking_factor, weather_speed_factor
from .models import (
    CrossingState,
    Journey,
    Signal,
    SignalAspect,
    SimulationConfig,
    StationStop,
    TelemetryFrame,
    TrainConfig,
    TrainDirection,
    WeatherCondition,
)
from .physics import kmh_to_ms, ms_to_kmh, safe_speed_for_target, update_position, update_velocity


@dataclass
class _Target:
    route_position_m: float
    speed_kmh: float
    reason: str
    hard_stop: bool = False
    station_id: str | None = None


@dataclass
class _RuntimeTrain:
    train: TrainConfig
    journey: Journey
    departure_time_s: float
    station_stops: list[StationStop]
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

    def active(self, sim_time_s: float) -> bool:
        return sim_time_s >= self.departure_time_s and not self.completed


class _Environment:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._weather = {item.block_id: item for item in config.environment.weather}
        self._signal_states = {item.signal_id: item for item in config.environment.signal_states}

    @staticmethod
    def active_entry(timeline, sim_time_s: float):
        active = timeline[0]
        for entry in timeline:
            if entry.start_time_s <= sim_time_s:
                active = entry
            else:
                break
        return active

    def weather(self, block_id: str, sim_time_s: float) -> tuple[WeatherCondition, float]:
        schedule = self._weather.get(block_id)
        if schedule is None:
            return WeatherCondition.CLEAR, 10000.0
        item = self.active_entry(schedule.timeline, sim_time_s)
        return item.condition, item.visibility_m

    def crossing_state(self, crossing, sim_time_s: float) -> CrossingState:
        return self.active_entry(crossing.timeline, sim_time_s).state

    def scheduled_signal(self, signal_id: str, sim_time_s: float) -> SignalAspect:
        schedule = self._signal_states.get(signal_id)
        if schedule is None:
            return SignalAspect.GREEN
        return self.active_entry(schedule.timeline, sim_time_s).aspect

    @staticmethod
    def restriction_active(restriction, sim_time_s: float) -> bool:
        return sim_time_s >= restriction.start_time_s and (
            restriction.end_time_s is None or sim_time_s < restriction.end_time_s
        )


class MultiTrainSimulationEngine:
    """Synchronous multi-train engine for parallel tracks.

    Trains share route-distance geometry but have independent track IDs and may
    travel in either direction. Dynamic signals are derived from same-track
    block occupancy. Stations introduce mandatory stop/dwell targets.
    """

    def __init__(self, config: SimulationConfig, scenario_id: str = "default"):
        self.config = config
        self.scenario_id = scenario_id
        self.env = _Environment(config)
        self.tick_count = 0
        self.sim_time_s = 0.0
        self.trains: list[_RuntimeTrain] = []

        primary_source = self._endpoint_m(config.journey.source)
        primary_destination = self._endpoint_m(config.journey.destination)
        self.trains.append(
            _RuntimeTrain(
                train=config.train,
                journey=config.journey,
                departure_time_s=0.0,
                station_stops=config.primary_station_stops,
                source_m=primary_source,
                destination_m=primary_destination,
                route_position_m=primary_source,
                speed_kmh=config.train.initial_state.initial_speed_kmh,
            )
        )
        for run in config.additional_train_runs:
            source = self._endpoint_m(run.journey.source)
            destination = self._endpoint_m(run.journey.destination)
            self.trains.append(
                _RuntimeTrain(
                    train=run.train,
                    journey=run.journey,
                    departure_time_s=run.departure_time_s,
                    station_stops=run.station_stops,
                    source_m=source,
                    destination_m=destination,
                    route_position_m=source,
                    speed_kmh=run.train.initial_state.initial_speed_kmh,
                )
            )

        self._signals = [(self._signal_position(s), s) for s in config.signals]

    def _endpoint_m(self, endpoint) -> float:
        return self.config.route.block_start_distance_m(endpoint.block_id) + endpoint.position_in_block_m

    def _signal_position(self, signal: Signal) -> float:
        start = self.config.route.block_start_distance_m(signal.protected_block_id)
        block = self.config.route.blocks[self.config.route.block_index(signal.protected_block_id)]
        return start if signal.direction == TrainDirection.FORWARD else start + block.length_m

    @property
    def is_complete(self) -> bool:
        return all(train.completed for train in self.trains)

    def _body_interval(self, train: _RuntimeTrain) -> tuple[float, float]:
        if train.direction == TrainDirection.FORWARD:
            return train.route_position_m - train.train.length_m, train.route_position_m
        return train.route_position_m, train.route_position_m + train.train.length_m

    def _block_occupied(self, track_id: str, block_index: int) -> bool:
        if block_index < 0 or block_index >= len(self.config.route.blocks):
            return False
        block = self.config.route.blocks[block_index]
        start = self.config.route.block_start_distance_m(block.block_id)
        end = start + block.length_m
        for train in self.trains:
            if train.train.track_id != track_id or train.completed or self.sim_time_s < train.departure_time_s:
                continue
            rear, front = self._body_interval(train)
            if front >= start - 1e-6 and rear <= end + 1e-6:
                return True
        return False

    def signal_aspect(self, signal: Signal) -> SignalAspect:
        if not self.config.dynamic_signalling:
            return self.env.scheduled_signal(signal.signal_id, self.sim_time_s)
        idx = self.config.route.block_index(signal.protected_block_id)
        if self._block_occupied(signal.track_id, idx):
            return SignalAspect.RED
        next_idx = idx + (1 if signal.direction == TrainDirection.FORWARD else -1)
        if self._block_occupied(signal.track_id, next_idx):
            return SignalAspect.YELLOW
        return SignalAspect.GREEN

    def signal_states(self) -> dict[str, SignalAspect]:
        return {signal.signal_id: self.signal_aspect(signal) for _, signal in self._signals}

    def _current_location(self, train: _RuntimeTrain):
        return self.config.route.locate(train.route_position_m)

    @staticmethod
    def _restriction_on_track(restriction, track_id: str) -> bool:
        return restriction.track_id is None or restriction.track_id == track_id

    def _current_ceiling(self, train: _RuntimeTrain) -> tuple[float, WeatherCondition, float]:
        _, block, local = self._current_location(train)
        weather, visibility = self.env.weather(block.block_id, self.sim_time_s)
        ceiling = min(train.train.max_speed_kmh, block.speed_limit_kmh)
        if block.curve_speed_limit_kmh is not None:
            ceiling = min(ceiling, block.curve_speed_limit_kmh)
        ceiling *= weather_speed_factor(weather, visibility)
        for restriction in [*self.config.environment.temporary_speed_restrictions, *self.config.environment.maintenance_restrictions]:
            if (
                self._restriction_on_track(restriction, train.train.track_id)
                and restriction.block_id == block.block_id
                and restriction.start_position_m <= local < restriction.end_position_m
                and self.env.restriction_active(restriction, self.sim_time_s)
            ):
                ceiling = min(ceiling, restriction.speed_limit_kmh)
        return max(0.0, ceiling), weather, visibility

    def _ahead_distance(self, train: _RuntimeTrain, target_m: float) -> float | None:
        distance = (target_m - train.route_position_m) * train.sign
        return distance if distance >= -1e-6 else None

    def _platform_position(self, station_id: str, track_id: str) -> float | None:
        station = next((item for item in self.config.stations if item.station_id == station_id), None)
        if station is None:
            return None
        platform = next((item for item in station.platforms if item.track_id == track_id), None)
        if platform is None:
            return None
        return self.config.route.block_start_distance_m(platform.block_id) + platform.position_in_block_m

    def _targets(self, train: _RuntimeTrain) -> list[_Target]:
        targets: list[_Target] = [
            _Target(train.destination_m, 0.0, "DESTINATION", hard_stop=True)
        ]

        for stop in train.station_stops:
            if stop.station_id in train.served_stations:
                continue
            pos = self._platform_position(stop.station_id, train.train.track_id)
            if pos is not None and self._ahead_distance(train, pos) is not None:
                targets.append(_Target(pos, 0.0, f"STATION:{stop.station_id}", hard_stop=True, station_id=stop.station_id))

        for crossing in self.config.environment.crossings:
            if self.env.crossing_state(crossing, self.sim_time_s) != CrossingState.CLOSED_FOR_TRAIN:
                continue
            pos = self.config.route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m
            if self._ahead_distance(train, pos) is not None:
                targets.append(_Target(pos, 0.0, f"CROSSING:{crossing.crossing_id}", hard_stop=True))

        for pos, signal in self._signals:
            if signal.track_id != train.train.track_id or signal.direction != train.direction:
                continue
            if self._ahead_distance(train, pos) is not None and self.signal_aspect(signal) == SignalAspect.RED:
                targets.append(_Target(pos, 0.0, f"RED_SIGNAL:{signal.signal_id}", hard_stop=True))

        # Direct separation target complements block signalling and prevents two
        # following trains from closing to an unrealistic distance within a block.
        separation = self.config.simulation.train_separation_m
        for other in self.trains:
            if other is train or other.train.track_id != train.train.track_id or other.completed:
                continue
            signed_gap = (other.route_position_m - train.route_position_m) * train.sign
            if signed_gap <= 0:
                continue
            if other.direction == train.direction:
                safe_front = other.route_position_m - train.sign * (other.train.length_m + separation)
                if self._ahead_distance(train, safe_front) is not None:
                    targets.append(_Target(safe_front, 0.0, f"TRAIN_AHEAD:{other.train.train_id}", hard_stop=True))

        # Upcoming block limits in either direction.
        idx, current_block, _ = self._current_location(train)
        indices = range(idx + 1, len(self.config.route.blocks)) if train.sign > 0 else range(idx - 1, -1, -1)
        for future_idx in indices:
            block = self.config.route.blocks[future_idx]
            start = self.config.route.block_start_distance_m(block.block_id)
            boundary = start if train.sign > 0 else start + block.length_m
            weather, visibility = self.env.weather(block.block_id, self.sim_time_s)
            limit = min(train.train.max_speed_kmh, block.speed_limit_kmh)
            if block.curve_speed_limit_kmh is not None:
                limit = min(limit, block.curve_speed_limit_kmh)
            limit *= weather_speed_factor(weather, visibility)
            if limit < current_block.speed_limit_kmh or block.curve_speed_limit_kmh is not None:
                targets.append(_Target(boundary, limit, f"UPCOMING_BLOCK:{block.block_id}"))

        return [target for target in targets if self._ahead_distance(train, target.route_position_m) is not None]

    def _desired_speed(self, train: _RuntimeTrain, ceiling: float, weather: WeatherCondition) -> tuple[float, str]:
        desired = ceiling
        reason = "CURRENT_SPEED_CEILING"
        _, block, _ = self._current_location(train)
        # Gradient sign reverses when travelling in the opposite direction.
        grade_component = 9.81 * (block.gradient_percent / 100.0) * train.sign
        decel = max(0.2, train.train.service_decel_ms2 * weather_braking_factor(weather) + grade_component)
        margin = self.config.simulation.braking_safety_margin_m
        for target in self._targets(train):
            distance = max(0.0, (target.route_position_m - train.route_position_m) * train.sign - margin)
            safe = ms_to_kmh(safe_speed_for_target(kmh_to_ms(target.speed_kmh), distance, decel))
            if safe < desired:
                desired = safe
                reason = target.reason
        return max(0.0, desired), reason

    def _nearest_hard_stop(self, train: _RuntimeTrain) -> Optional[_Target]:
        stops = [t for t in self._targets(train) if t.hard_stop]
        return min(stops, key=lambda t: (t.route_position_m - train.route_position_m) * train.sign) if stops else None

    def _start_station_dwell_if_needed(self, train: _RuntimeTrain, target: _Target) -> None:
        if target.station_id is None:
            return
        stop = next((item for item in train.station_stops if item.station_id == target.station_id), None)
        if stop is None:
            return
        train.dwelling_station_id = target.station_id
        train.dwell_until_s = self.sim_time_s + stop.dwell_time_s

    def _advance_train(self, train: _RuntimeTrain, dt: float) -> tuple[str, str]:
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
        tol = self.config.simulation.speed_tolerance_kmh
        _, block, _ = self._current_location(train)
        grade_component = 9.81 * (block.gradient_percent / 100.0) * train.sign
        if train.speed_kmh > desired + tol:
            accel = -max(0.2, train.train.service_decel_ms2 * weather_braking_factor(weather) + grade_component)
            action = "BRAKE"
        elif train.speed_kmh < desired - tol:
            accel = max(0.05, train.train.accel_ms2 - grade_component)
            action = "ACCELERATE"
        else:
            accel = 0.0
            action = "MAINTAIN"

        old_v = kmh_to_ms(train.speed_kmh)
        desired_v = kmh_to_ms(desired)
        new_v = update_velocity(old_v, accel, dt)
        if accel > 0:
            new_v = min(new_v, desired_v)
        elif accel < 0:
            new_v = max(new_v, desired_v)
        travelled = max(0.0, update_position(0.0, old_v, accel, dt))
        new_pos = train.route_position_m + train.sign * travelled

        hard_stop = self._nearest_hard_stop(train)
        if hard_stop is not None:
            passed = (new_pos - hard_stop.route_position_m) * train.sign >= 0
            if passed:
                if hard_stop.reason == "DESTINATION":
                    new_pos = hard_stop.route_position_m
                    new_v = 0.0
                    train.completed = True
                    action, reason = "STOP", "DESTINATION"
                else:
                    new_pos = hard_stop.route_position_m - train.sign * 0.01
                    new_v = 0.0
                    action, reason = "STOP", hard_stop.reason
                    if hard_stop.station_id:
                        new_pos = hard_stop.route_position_m
                        self._start_station_dwell_if_needed(train, hard_stop)

        train.route_position_m = max(0.0, min(self.config.route.total_length_m, new_pos))
        train.speed_kmh = ms_to_kmh(new_v)
        train.acceleration_ms2 = 0.0 if train.completed else accel
        return action, reason

    def tick(self) -> list[TelemetryFrame]:
        if self.is_complete:
            return []
        dt = self.config.simulation.tick_seconds
        actions: dict[str, tuple[str, str]] = {}
        for train in self.trains:
            actions[train.train.train_id] = self._advance_train(train, dt)
        self.tick_count += 1
        self.sim_time_s += dt
        return [self.snapshot_train(train, *actions[train.train.train_id]) for train in self.trains]

    def snapshot_all(self) -> list[TelemetryFrame]:
        return [self.snapshot_train(train) for train in self.trains]

    def snapshot_train(self, train: _RuntimeTrain, action: str = "INITIAL", reason: str = "INITIAL_STATE") -> TelemetryFrame:
        idx, block, local = self._current_location(train)
        ceiling, weather, visibility = self._current_ceiling(train)
        distance = max(0.0, (train.destination_m - train.route_position_m) * train.sign)
        journey_length = abs(train.destination_m - train.source_m)
        progress = min(1.0, max(0.0, abs(train.route_position_m - train.source_m) / max(1e-6, journey_length)))

        signals = []
        for pos, signal in self._signals:
            if signal.track_id != train.train.track_id or signal.direction != train.direction:
                continue
            d = self._ahead_distance(train, pos)
            if d is not None and d > 1e-6:
                signals.append((d, signal, self.signal_aspect(signal)))
        signals.sort(key=lambda item: item[0])

        next_speed = next_speed_distance = next_curve = next_curve_distance = None
        indices = range(idx + 1, len(self.config.route.blocks)) if train.sign > 0 else range(idx - 1, -1, -1)
        for future_idx in indices:
            future = self.config.route.blocks[future_idx]
            start = self.config.route.block_start_distance_m(future.block_id)
            boundary = start if train.sign > 0 else start + future.length_m
            d = max(0.0, (boundary - train.route_position_m) * train.sign)
            if next_speed is None and future.speed_limit_kmh != block.speed_limit_kmh:
                next_speed, next_speed_distance = future.speed_limit_kmh, d
            if next_curve is None and future.curve_speed_limit_kmh is not None:
                next_curve, next_curve_distance = future.curve_speed_limit_kmh, d
            if next_speed is not None and next_curve is not None:
                break

        next_tsr = None
        for restriction in self.config.environment.temporary_speed_restrictions:
            if (
                not self._restriction_on_track(restriction, train.train.track_id)
                or not self.env.restriction_active(restriction, self.sim_time_s)
            ):
                continue
            block_start = self.config.route.block_start_distance_m(restriction.block_id)
            target = block_start + (restriction.start_position_m if train.sign > 0 else restriction.end_position_m)
            d = self._ahead_distance(train, target)
            if d is not None and (next_tsr is None or d < next_tsr[0]):
                next_tsr = (d, restriction.speed_limit_kmh)

        next_crossing = None
        for crossing in self.config.environment.crossings:
            pos = self.config.route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m
            d = self._ahead_distance(train, pos)
            if d is not None and (next_crossing is None or d < next_crossing[0]):
                next_crossing = (d, self.env.crossing_state(crossing, self.sim_time_s))

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
            next_signal_aspect=signals[0][2] if len(signals) > 0 else None,
            distance_to_next_signal_m=signals[0][0] if len(signals) > 0 else None,
            second_signal_aspect=signals[1][2] if len(signals) > 1 else None,
            distance_to_second_signal_m=signals[1][0] if len(signals) > 1 else None,
            next_speed_limit_kmh=next_speed,
            distance_to_next_speed_change_m=next_speed_distance,
            next_curve_limit_kmh=next_curve,
            distance_to_next_curve_m=next_curve_distance,
            next_tsr_limit_kmh=next_tsr[1] if next_tsr else None,
            distance_to_next_tsr_m=next_tsr[0] if next_tsr else None,
            next_crossing_state=next_crossing[1] if next_crossing else None,
            distance_to_next_crossing_m=next_crossing[0] if next_crossing else None,
        )

    def run(self) -> list[TelemetryFrame]:
        frames = self.snapshot_all()
        while not self.is_complete and self.sim_time_s < self.config.simulation.max_simulation_time_s:
            frames.extend(self.tick())
        if not self.is_complete:
            raise RuntimeError(f"Scenario {self.scenario_id} timed out at {self.sim_time_s:.1f}s")
        return frames
