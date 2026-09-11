from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .guardrails import weather_braking_factor, weather_speed_factor
from .models import CrossingState, SimulationConfig, SignalAspect, TelemetryFrame, WeatherCondition
from .physics import kmh_to_ms, ms_to_kmh, safe_speed_for_target, update_position, update_velocity


@dataclass
class _SpeedTarget:
    route_position_m: float
    target_speed_kmh: float
    reason: str
    hard_stop: bool = False


class TrainState:
    def __init__(self, config: SimulationConfig):
        initial = config.train.initial_state
        self.route_position_m = config.route.block_start_distance_m(initial.start_block_id) + initial.position_in_block_m
        self.speed_kmh = initial.initial_speed_kmh
        self.acceleration_ms2 = 0.0
        self.completed = False


class EnvironmentResolver:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._weather = {x.block_id: x for x in config.environment.weather}
        self._signal_states = {x.signal_id: x for x in config.environment.signal_states}

    @staticmethod
    def _active_entry(timeline, sim_time_s: float):
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
        entry = self._active_entry(schedule.timeline, sim_time_s)
        return entry.condition, entry.visibility_m

    def signal_aspect(self, signal_id: str, sim_time_s: float) -> SignalAspect:
        schedule = self._signal_states.get(signal_id)
        if schedule is None:
            return SignalAspect.GREEN
        return self._active_entry(schedule.timeline, sim_time_s).aspect

    def crossing_state(self, crossing, sim_time_s: float) -> CrossingState:
        return self._active_entry(crossing.timeline, sim_time_s).state

    @staticmethod
    def restriction_active(restriction, sim_time_s: float) -> bool:
        return sim_time_s >= restriction.start_time_s and (
            restriction.end_time_s is None or sim_time_s < restriction.end_time_s
        )


class SimulationEngine:
    """Single-train source-to-destination simulator for Component A."""

    def __init__(self, config: SimulationConfig, scenario_id: str = "default"):
        self.config = config
        self.scenario_id = scenario_id
        self.state = TrainState(config)
        self.env = EnvironmentResolver(config)
        self.tick_count = 0
        self.sim_time_s = 0.0
        self.source_route_m = self._endpoint_route_m(config.journey.source)
        self.destination_route_m = self._endpoint_route_m(config.journey.destination)
        if self.state.route_position_m < self.source_route_m:
            self.state.route_position_m = self.source_route_m
        self._signal_positions = sorted(
            [(config.route.block_start_distance_m(s.protected_block_id), s) for s in config.signals],
            key=lambda item: item[0],
        )

    @property
    def is_complete(self) -> bool:
        return self.state.completed

    def _endpoint_route_m(self, endpoint) -> float:
        return self.config.route.block_start_distance_m(endpoint.block_id) + endpoint.position_in_block_m

    def _current_location(self):
        return self.config.route.locate(self.state.route_position_m)

    def _current_ceiling(self, sim_time_s: float) -> tuple[float, WeatherCondition, float]:
        _, block, pos_in_block = self._current_location()
        weather, visibility = self.env.weather(block.block_id, sim_time_s)
        ceiling = min(self.config.train.max_speed_kmh, block.speed_limit_kmh)
        if block.curve_speed_limit_kmh is not None:
            ceiling = min(ceiling, block.curve_speed_limit_kmh)
        ceiling *= weather_speed_factor(weather, visibility)
        for restriction in [*self.config.environment.temporary_speed_restrictions, *self.config.environment.maintenance_restrictions]:
            if (
                restriction.block_id == block.block_id
                and restriction.start_position_m <= pos_in_block < restriction.end_position_m
                and self.env.restriction_active(restriction, sim_time_s)
            ):
                ceiling = min(ceiling, restriction.speed_limit_kmh)
        return max(0.0, ceiling), weather, visibility

    def _targets_ahead(self, sim_time_s: float) -> list[_SpeedTarget]:
        pos = self.state.route_position_m
        route = self.config.route
        targets: list[_SpeedTarget] = []
        current_idx, _, _ = self._current_location()

        for block in route.blocks[current_idx + 1:]:
            boundary = route.block_start_distance_m(block.block_id)
            weather, visibility = self.env.weather(block.block_id, sim_time_s)
            limit = min(self.config.train.max_speed_kmh, block.speed_limit_kmh)
            if block.curve_speed_limit_kmh is not None:
                limit = min(limit, block.curve_speed_limit_kmh)
            limit *= weather_speed_factor(weather, visibility)
            targets.append(_SpeedTarget(boundary, limit, f"UPCOMING_BLOCK:{block.block_id}"))

        for restriction in [*self.config.environment.temporary_speed_restrictions, *self.config.environment.maintenance_restrictions]:
            if not self.env.restriction_active(restriction, sim_time_s):
                continue
            start = route.block_start_distance_m(restriction.block_id) + restriction.start_position_m
            if start > pos:
                targets.append(_SpeedTarget(start, restriction.speed_limit_kmh, f"RESTRICTION:{restriction.restriction_id}"))

        for signal_pos, signal in self._signal_positions:
            if signal_pos < pos - 1e-6:
                continue
            if self.env.signal_aspect(signal.signal_id, sim_time_s) == SignalAspect.RED:
                targets.append(_SpeedTarget(signal_pos, 0.0, f"RED_SIGNAL:{signal.signal_id}", hard_stop=True))

        for crossing in self.config.environment.crossings:
            crossing_pos = route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m
            if crossing_pos < pos - 1e-6:
                continue
            if self.env.crossing_state(crossing, sim_time_s) == CrossingState.CLOSED_FOR_TRAIN:
                targets.append(_SpeedTarget(crossing_pos, 0.0, f"CROSSING:{crossing.crossing_id}", hard_stop=True))

        targets.append(_SpeedTarget(self.destination_route_m, 0.0, "DESTINATION", hard_stop=True))
        return [t for t in targets if t.route_position_m >= pos - 1e-6]

    def _desired_speed(self, sim_time_s: float, current_ceiling_kmh: float, weather: WeatherCondition) -> tuple[float, str]:
        desired = current_ceiling_kmh
        reason = "CURRENT_SPEED_CEILING"
        _, block, _ = self._current_location()
        grade_component = 9.81 * (block.gradient_percent / 100.0)
        decel = max(0.2, self.config.train.service_decel_ms2 * weather_braking_factor(weather) + grade_component)
        margin = self.config.simulation.braking_safety_margin_m
        current_pos = self.state.route_position_m
        for target in self._targets_ahead(sim_time_s):
            distance = max(0.0, target.route_position_m - current_pos - margin)
            safe_kmh = ms_to_kmh(safe_speed_for_target(kmh_to_ms(target.target_speed_kmh), distance, decel))
            if safe_kmh < desired:
                desired = safe_kmh
                reason = target.reason
        return max(0.0, desired), reason

    def _nearest_hard_stop(self, sim_time_s: float) -> Optional[_SpeedTarget]:
        hard_stops = [t for t in self._targets_ahead(sim_time_s) if t.hard_stop]
        return min(hard_stops, key=lambda t: t.route_position_m) if hard_stops else None

    def _control(self, desired_speed_kmh: float, reason: str, weather: WeatherCondition) -> tuple[float, str, str]:
        speed = self.state.speed_kmh
        tol = self.config.simulation.speed_tolerance_kmh
        _, block, _ = self._current_location()
        grade_component = 9.81 * (block.gradient_percent / 100.0)
        if speed > desired_speed_kmh + tol:
            decel = max(0.2, self.config.train.service_decel_ms2 * weather_braking_factor(weather) + grade_component)
            return -decel, "BRAKE", reason
        if speed < desired_speed_kmh - tol:
            accel = max(0.05, self.config.train.accel_ms2 - grade_component)
            return accel, "ACCELERATE", reason
        return 0.0, "MAINTAIN", reason

    def tick(self) -> list[TelemetryFrame]:
        if self.state.completed:
            return []
        dt = self.config.simulation.tick_seconds
        current_ceiling, weather, _ = self._current_ceiling(self.sim_time_s)
        desired_speed, reason = self._desired_speed(self.sim_time_s, current_ceiling, weather)
        acceleration, action, reason = self._control(desired_speed, reason, weather)

        old_v_ms = kmh_to_ms(self.state.speed_kmh)
        desired_ms = kmh_to_ms(desired_speed)
        new_v_ms = update_velocity(old_v_ms, acceleration, dt)
        if acceleration > 0:
            new_v_ms = min(new_v_ms, desired_ms)
        elif acceleration < 0:
            new_v_ms = max(new_v_ms, desired_ms)
        new_pos = max(self.state.route_position_m, update_position(self.state.route_position_m, old_v_ms, acceleration, dt))

        hard_stop = self._nearest_hard_stop(self.sim_time_s)
        if hard_stop is not None and hard_stop.reason != "DESTINATION" and new_pos >= hard_stop.route_position_m:
            new_pos = max(self.state.route_position_m, hard_stop.route_position_m - 0.01)
            new_v_ms = 0.0
            action = "STOP"
            reason = hard_stop.reason

        if new_pos >= self.destination_route_m:
            new_pos = self.destination_route_m
            new_v_ms = 0.0
            self.state.completed = True
            action = "STOP"
            reason = "DESTINATION"

        self.state.route_position_m = new_pos
        self.state.speed_kmh = ms_to_kmh(new_v_ms)
        self.state.acceleration_ms2 = acceleration if not self.state.completed else 0.0
        self.tick_count += 1
        self.sim_time_s += dt
        return [self.snapshot(action=action, reason=reason)]

    def snapshot(self, action: str = "INITIAL", reason: str = "INITIAL_STATE") -> TelemetryFrame:
        idx, block, pos_in_block = self._current_location()
        current_ceiling, weather, visibility = self._current_ceiling(self.sim_time_s)
        route_pos = self.state.route_position_m
        distance_to_destination = max(0.0, self.destination_route_m - route_pos)
        journey_length = self.destination_route_m - self.source_route_m
        progress = min(1.0, max(0.0, (route_pos - self.source_route_m) / journey_length))

        next_signals = []
        for signal_pos, signal in self._signal_positions:
            if signal_pos > route_pos + 1e-6:
                next_signals.append((signal_pos, signal, self.env.signal_aspect(signal.signal_id, self.sim_time_s)))

        next_speed_limit = next_speed_distance = next_curve_limit = next_curve_distance = None
        for future_block in self.config.route.blocks[idx + 1:]:
            boundary = self.config.route.block_start_distance_m(future_block.block_id)
            if next_speed_limit is None and future_block.speed_limit_kmh != block.speed_limit_kmh:
                next_speed_limit, next_speed_distance = future_block.speed_limit_kmh, boundary - route_pos
            if next_curve_limit is None and future_block.curve_speed_limit_kmh is not None:
                next_curve_limit, next_curve_distance = future_block.curve_speed_limit_kmh, boundary - route_pos
            if next_speed_limit is not None and next_curve_limit is not None:
                break

        next_tsr = None
        for restriction in self.config.environment.temporary_speed_restrictions:
            if not self.env.restriction_active(restriction, self.sim_time_s):
                continue
            start = self.config.route.block_start_distance_m(restriction.block_id) + restriction.start_position_m
            if start > route_pos:
                candidate = (start - route_pos, restriction.speed_limit_kmh)
                if next_tsr is None or candidate[0] < next_tsr[0]:
                    next_tsr = candidate

        next_crossing = None
        for crossing in self.config.environment.crossings:
            cpos = self.config.route.block_start_distance_m(crossing.block_id) + crossing.position_in_block_m
            if cpos > route_pos:
                candidate = (cpos - route_pos, self.env.crossing_state(crossing, self.sim_time_s))
                if next_crossing is None or candidate[0] < next_crossing[0]:
                    next_crossing = candidate

        return TelemetryFrame(
            scenario_id=self.scenario_id,
            train_id=self.config.train.train_id,
            tick=self.tick_count,
            sim_time_s=self.sim_time_s,
            current_block_id=block.block_id,
            position_in_block_m=pos_in_block,
            route_position_m=route_pos,
            speed_kmh=self.state.speed_kmh,
            acceleration_ms2=self.state.acceleration_ms2,
            control_action=action,
            control_reason=reason,
            current_block_speed_limit_kmh=block.speed_limit_kmh,
            current_gradient_percent=block.gradient_percent,
            current_curve_speed_limit_kmh=block.curve_speed_limit_kmh,
            effective_speed_ceiling_kmh=current_ceiling,
            weather=weather,
            visibility_m=visibility,
            distance_to_destination_m=distance_to_destination,
            route_progress=progress,
            next_signal_aspect=next_signals[0][2] if len(next_signals) > 0 else None,
            distance_to_next_signal_m=(next_signals[0][0] - route_pos) if len(next_signals) > 0 else None,
            second_signal_aspect=next_signals[1][2] if len(next_signals) > 1 else None,
            distance_to_second_signal_m=(next_signals[1][0] - route_pos) if len(next_signals) > 1 else None,
            next_speed_limit_kmh=next_speed_limit,
            distance_to_next_speed_change_m=next_speed_distance,
            next_curve_limit_kmh=next_curve_limit,
            distance_to_next_curve_m=next_curve_distance,
            next_tsr_limit_kmh=next_tsr[1] if next_tsr else None,
            distance_to_next_tsr_m=next_tsr[0] if next_tsr else None,
            next_crossing_state=next_crossing[1] if next_crossing else None,
            distance_to_next_crossing_m=next_crossing[0] if next_crossing else None,
        )

    def run(self) -> list[TelemetryFrame]:
        frames = [self.snapshot()]
        while not self.state.completed and self.sim_time_s < self.config.simulation.max_simulation_time_s:
            frames.extend(self.tick())
        if not self.state.completed:
            raise RuntimeError(f"Scenario {self.scenario_id} timed out at {self.sim_time_s:.1f}s before destination")
        return frames
