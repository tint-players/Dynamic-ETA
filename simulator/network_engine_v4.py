from __future__ import annotations

from .guardrails import weather_braking_factor
from .models import CrossingState, SignalAspect, SignalType
from .network_engine import RuntimeTrain, Target
from .network_engine_v3 import NetworkSimulationEngineV3
from .physics import kmh_to_ms, ms_to_kmh, update_velocity


class NetworkSimulationEngineV4(NetworkSimulationEngineV3):
    """Network engine with safe crossovers, turnarounds and train-triggered level crossings.

    Safety rules added in v4:
    - A train body, not just its front, determines track occupancy.
    - A train straddling a crossover occupies both connected tracks until its rear
      has cleared the crossover, even after runtime track_id switches.
    - Planned crossover moves reserve the conflict zone before entry. Traffic on
      either connected track is held outside the zone until the owner clears it.
    - A train is removed from active-network occupancy when its journey completes;
      the fleet still retains its completed telemetry record.
    - Level crossings are train-triggered and use a stop line before the road.
    - RED railway signals use a protected stop line before the signal post.

    Level crossings are controlled from the road-user perspective:
    - road open => CLOSED_FOR_TRAIN
    - road closed/protected => OPEN_FOR_TRAIN
    """

    CROSSING_APPROACH_TRIGGER_M = 600.0
    CROSSING_STOP_LINE_M = 60.0
    CROSSING_GATE_CLOSING_S = 12.0
    CROSSING_CLEARANCE_MARGIN_M = 20.0
    CROSSING_REOPEN_DELAY_S = 5.0

    # Front of train must stop this far before a RED signal post. This is a
    # simulator protection margin, not a claim about a railway-standard value.
    SIGNAL_STOP_MARGIN_M = 20.0

    CROSSOVER_RESERVATION_APPROACH_M = 650.0
    CROSSOVER_STOP_MARGIN_M = 70.0

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
        self._crossover_reservations: dict[str, str | None] = {
            crossover.crossover_id: None for crossover in self.config.crossovers
        }
        self._update_crossings()
        self._update_crossover_reservations()

    def _signal_position(self, signal) -> float:
        if signal.position_in_block_m is not None:
            return self.config.route.block_start_distance_m(signal.protected_block_id) + signal.position_in_block_m
        return super()._signal_position(signal)

    def signal_aspect(self, signal, exclude: RuntimeTrain | None = None) -> SignalAspect:
        if signal.signal_type != SignalType.ROUTE_INDICATOR or signal.crossover_id is None:
            return super().signal_aspect(signal, exclude=exclude)

        owner_id = self._crossover_reservations.get(signal.crossover_id)
        if owner_id is None:
            return SignalAspect.GREEN

        crossover = self._crossovers[signal.crossover_id]
        owner = next((train for train in self.trains if train.train.train_id == owner_id), None)
        if owner is None:
            return SignalAspect.RED
        plan = self._train_plan_for_crossover(owner, signal.crossover_id)
        owner_is_entering_route = (
            plan is not None
            and signal.crossover_id not in owner.completed_crossovers
            and owner.current_track_id == crossover.from_track_id
            and signal.track_id == crossover.from_track_id
            and signal.direction == owner.direction
        )
        return SignalAspect.GREEN if owner_is_entering_route else SignalAspect.RED

    # ------------------------------------------------------------------
    # Shared physical occupancy / train separation
    # ------------------------------------------------------------------
    def _body_bounds(self, train: RuntimeTrain) -> tuple[float, float]:
        a, b = self._body_interval(train)
        return min(a, b), max(a, b)

    @staticmethod
    def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
        return a1 >= b0 - 1e-6 and b1 >= a0 - 1e-6

    def _occupancy_tracks(self, train: RuntimeTrain) -> set[str]:
        """Return every track physically occupied by any part of the train body."""
        tracks = {train.current_track_id}
        body_start, body_end = self._body_bounds(train)
        for plan in train.track_changes:
            crossover = self._crossovers[plan.crossover_id]
            start, end, _ = self._crossover_bounds(crossover)
            if self._intervals_overlap(body_start, body_end, start, end):
                tracks.update({crossover.from_track_id, crossover.to_track_id})
        return tracks

    def _block_occupied(self, track_id: str, block_index: int, exclude: RuntimeTrain | None = None) -> bool:
        if block_index < 0 or block_index >= len(self.config.route.blocks):
            return False
        block = self.config.route.blocks[block_index]
        start = self.config.route.block_start_distance_m(block.block_id)
        end = start + block.length_m
        for train in self.trains:
            # completed means the train has exited this simulated corridor; it is
            # kept in telemetry/fleet history but no longer occupies active track.
            if train is exclude or train.completed or self.sim_time_s < train.departure_time_s:
                continue
            if track_id not in self._occupancy_tracks(train):
                continue
            body_start, body_end = self._body_bounds(train)
            if self._intervals_overlap(body_start, body_end, start, end):
                return True
        return False

    def _separation_targets(self, train: RuntimeTrain) -> list[Target]:
        targets: list[Target] = []
        separation = self.config.simulation.train_separation_m
        own_tracks = self._occupancy_tracks(train)

        for other in self.trains:
            if other is train or other.completed or self.sim_time_s < other.departure_time_s:
                continue
            if not own_tracks.intersection(self._occupancy_tracks(other)):
                continue

            gap = (other.route_position_m - train.route_position_m) * train.sign
            if gap <= 0:
                continue

            if other.direction == train.direction:
                safe_pos = other.route_position_m - train.sign * (other.train.length_m + separation)
            else:
                safe_pos = other.route_position_m - train.sign * separation

            if self._ahead(train, safe_pos) is not None:
                targets.append(Target(safe_pos, 0.0, f"TRAIN_AHEAD:{other.train.train_id}", True))
        return targets

    # ------------------------------------------------------------------
    # Crossover interlocking / reservation
    # ------------------------------------------------------------------
    def _train_plan_for_crossover(self, train: RuntimeTrain, crossover_id: str):
        return next((plan for plan in train.track_changes if plan.crossover_id == crossover_id), None)

    def _train_body_in_crossover(self, train: RuntimeTrain, crossover) -> bool:
        start, end, _ = self._crossover_bounds(crossover)
        body_start, body_end = self._body_bounds(train)
        return self._intervals_overlap(body_start, body_end, start, end)

    def _update_crossover_reservations(self) -> None:
        for crossover in self.config.crossovers:
            cid = crossover.crossover_id
            owner_id = self._crossover_reservations.get(cid)
            start, end, _ = self._crossover_bounds(crossover)

            # First decide whether an existing owner still legitimately holds the
            # interlocking. This supports both planned crossover users and normal
            # trains that happened to already occupy one of the parallel tracks.
            if owner_id is not None:
                owner = next((t for t in self.trains if t.train.train_id == owner_id), None)
                keep = False
                if owner is not None and not owner.completed and self.sim_time_s >= owner.departure_time_s:
                    if self._train_body_in_crossover(owner, crossover):
                        keep = True
                    else:
                        plan = self._train_plan_for_crossover(owner, cid)
                        if plan is not None and cid not in owner.completed_crossovers and owner.current_track_id == crossover.from_track_id:
                            entry = start if owner.sign > 0 else end
                            distance = (entry - owner.route_position_m) * owner.sign
                            keep = 0.0 <= distance <= self.CROSSOVER_RESERVATION_APPROACH_M
                if not keep:
                    self._crossover_reservations[cid] = None
                    owner_id = None

            if owner_id is not None:
                continue

            # An already occupied conflict zone always gets priority. This prevents
            # assigning a switch movement across a train that is already passing on
            # one of the connected tracks.
            occupants: list[RuntimeTrain] = []
            connected = {crossover.from_track_id, crossover.to_track_id}
            for train in self.trains:
                if train.completed or self.sim_time_s < train.departure_time_s:
                    continue
                if train.current_track_id not in connected:
                    continue
                if self._train_body_in_crossover(train, crossover):
                    occupants.append(train)
            if occupants:
                self._crossover_reservations[cid] = occupants[0].train.train_id
                continue

            # Otherwise reserve ahead of time for the nearest train that actually
            # has a route plan to change track at this crossover.
            candidates: list[tuple[float, RuntimeTrain]] = []
            for train in self.trains:
                if train.completed or self.sim_time_s < train.departure_time_s:
                    continue
                plan = self._train_plan_for_crossover(train, cid)
                if plan is None or cid in train.completed_crossovers:
                    continue
                if train.current_track_id != crossover.from_track_id:
                    continue
                entry = start if train.sign > 0 else end
                distance = (entry - train.route_position_m) * train.sign
                if 0.0 <= distance <= self.CROSSOVER_RESERVATION_APPROACH_M:
                    candidates.append((distance, train))

            if candidates:
                candidates.sort(key=lambda item: item[0])
                self._crossover_reservations[cid] = candidates[0][1].train.train_id

    def crossover_reservations(self) -> dict[str, str | None]:
        return dict(self._crossover_reservations)

    def _crossover_hold_targets(self, train: RuntimeTrain) -> list[Target]:
        targets: list[Target] = []
        for crossover in self.config.crossovers:
            owner_id = self._crossover_reservations.get(crossover.crossover_id)
            if owner_id is None or owner_id == train.train.train_id:
                continue
            if train.current_track_id not in {crossover.from_track_id, crossover.to_track_id}:
                continue

            start, end, _ = self._crossover_bounds(crossover)
            if train.sign > 0:
                stop_pos = start - self.CROSSOVER_STOP_MARGIN_M
                zone_ahead = self._ahead(train, start)
            else:
                stop_pos = end + self.CROSSOVER_STOP_MARGIN_M
                zone_ahead = self._ahead(train, end)

            if zone_ahead is None:
                continue
            # If reservation is established unusually late, stop at the current
            # front position rather than allowing the train to enter the zone.
            if self._ahead(train, stop_pos) is None:
                stop_pos = train.route_position_m
            targets.append(Target(stop_pos, 0.0, f"CROSSOVER_RESERVED:{crossover.crossover_id}", True))
        return targets

    def _pending_turnaround_plan(self, train: RuntimeTrain):
        crossover_id = getattr(train, "pending_turnaround_crossover_id", None)
        if crossover_id is None:
            return None
        return self._train_plan_for_crossover(train, crossover_id)

    def _turnaround_stop_position(self, train: RuntimeTrain, plan) -> float | None:
        if plan is None or plan.turnaround_signal_id is None:
            return None
        signal = next((item for item in self.config.signals if item.signal_id == plan.turnaround_signal_id), None)
        if signal is None:
            return None
        return self._signal_position(signal) - train.sign * self.SIGNAL_STOP_MARGIN_M

    def _begin_pending_turnaround_if_stopped(self, train: RuntimeTrain) -> str | None:
        plan = self._pending_turnaround_plan(train)
        stop_position = self._turnaround_stop_position(train, plan)
        if plan is None or stop_position is None:
            return None
        if abs(train.route_position_m - stop_position) > 0.05 or train.speed_kmh > 0.05:
            return None

        original_source = train.source_m
        train.source_m = train.route_position_m
        train.destination_m = original_source
        train.pending_turnaround_crossover_id = None
        return f"TURNAROUND:{plan.crossover_id}"

    # ------------------------------------------------------------------
    # Train-triggered level crossings
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Unified target builder
    # ------------------------------------------------------------------
    def _targets(self, train: RuntimeTrain) -> list[Target]:
        targets = [
            target for target in super()._targets(train)
            if not target.reason.startswith("TRAIN_AHEAD:")
        ]

        adjusted: list[Target] = []
        for target in targets:
            if target.reason.startswith("RED_SIGNAL:"):
                signal_position = target.position_m
                stop_line = signal_position - train.sign * self.SIGNAL_STOP_MARGIN_M
                # If the signal became RED after the train already entered the
                # margin, stop immediately rather than allowing its front to reach
                # or pass the signal post.
                if self._ahead(train, stop_line) is None and self._ahead(train, signal_position) is not None:
                    stop_line = train.route_position_m
                adjusted.append(Target(stop_line, 0.0, target.reason, True))
                continue

            if target.reason.startswith("CROSSING:"):
                crossing_id = target.reason.split(":", 1)[1]
                crossing = next(item for item in self.config.environment.crossings if item.crossing_id == crossing_id)
                crossing_pos = self._crossing_position(crossing)
                stop_line = crossing_pos - train.sign * self.CROSSING_STOP_LINE_M
                if self._ahead(train, stop_line) is None and self._ahead(train, crossing_pos) is not None:
                    stop_line = train.route_position_m
                adjusted.append(Target(stop_line, 0.0, target.reason, True))
                continue

            adjusted.append(target)

        turnaround_plan = self._pending_turnaround_plan(train)
        turnaround_stop = self._turnaround_stop_position(train, turnaround_plan)
        if turnaround_plan is not None and turnaround_stop is not None and self._ahead(train, turnaround_stop) is not None:
            adjusted.append(Target(
                turnaround_stop,
                0.0,
                f"TURNAROUND_STOP:{turnaround_plan.crossover_id}:{turnaround_plan.turnaround_signal_id}",
                True,
            ))

        adjusted.extend(self._separation_targets(train))
        adjusted.extend(self._crossover_hold_targets(train))
        return [target for target in adjusted if self._ahead(train, target.position_m) is not None]

    # ------------------------------------------------------------------
    # Crossover execution / turnaround
    # ------------------------------------------------------------------
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
                train.pending_turnaround_crossover_id = plan.crossover_id
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

        turnaround_reason = self._begin_pending_turnaround_if_stopped(train)

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
        elif turnaround_reason and action != "STOP":
            reason = turnaround_reason

        train.speed_kmh = ms_to_kmh(new_v)
        train.acceleration_ms2 = 0.0 if train.completed or turned_around else accel
        return action, reason

    def tick(self):
        if self.is_complete:
            return []
        self._update_crossings()
        self._update_crossover_reservations()
        dt = self.config.simulation.tick_seconds
        actions = {train.train.train_id: self._advance(train, dt) for train in self.trains}
        self.tick_count += 1
        self.sim_time_s += dt
        self._update_crossings()
        self._update_crossover_reservations()
        return [self.snapshot_train(train, *actions[train.train.train_id]) for train in self.trains]
