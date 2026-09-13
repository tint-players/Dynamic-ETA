from __future__ import annotations

from .models import SignalAspect, SignalType, WeatherCondition
from .network_engine import RuntimeTrain, Target
from .network_engine_v4 import NetworkSimulationEngineV4


class NetworkSimulationEngineV4Restrictive(NetworkSimulationEngineV4):
    """V4 network engine with conservative restrictive-signal approach control.

    The existing v4 RED stop targets, crossing protection, separation and
    crossover interlocking remain unchanged. This layer prevents a train from
    accelerating back toward line speed while the governing signal is YELLOW,
    keeps that caution speed after passing a YELLOW while the next signal remains
    restrictive, preserves configured station-stop order for turnaround trains,
    adds a conservative station-entry hold when the required platform is
    physically occupied by another active train, and supports temporary manual
    signal overrides for Phase-2 anomaly injection.
    """

    YELLOW_APPROACH_SPEED_KMH = 60.0
    STATION_ENTRY_MARGIN_M = 20.0

    def __init__(self, config, scenario_id: str = "default"):
        super().__init__(config, scenario_id=scenario_id)
        self._manual_signal_overrides: dict[str, tuple[SignalAspect, float | None]] = {}

    def set_manual_signal_override(
        self,
        signal_id: str,
        aspect: SignalAspect,
        duration_s: float | None,
    ) -> None:
        if not any(signal.signal_id == signal_id for signal in self.config.signals):
            raise ValueError(f"Unknown signal_id: {signal_id}")
        expires_at = self.sim_time_s + duration_s if duration_s is not None else None
        self._manual_signal_overrides[signal_id] = (aspect, expires_at)

    def manual_signal_override_details(self) -> dict[str, tuple[SignalAspect, float | None]]:
        expired = [
            signal_id
            for signal_id, (_, expires_at) in self._manual_signal_overrides.items()
            if expires_at is not None and self.sim_time_s >= expires_at
        ]
        for signal_id in expired:
            self._manual_signal_overrides.pop(signal_id, None)
        return dict(self._manual_signal_overrides)

    def manual_signal_overrides(self) -> dict[str, SignalAspect]:
        return {
            signal_id: aspect
            for signal_id, (aspect, _) in self.manual_signal_override_details().items()
        }

    def clear_manual_signal_override(self, signal_id: str) -> None:
        if not any(signal.signal_id == signal_id for signal in self.config.signals):
            raise ValueError(f"Unknown signal_id: {signal_id}")
        self._manual_signal_overrides.pop(signal_id, None)

    def clear_manual_signal_overrides(self) -> None:
        self._manual_signal_overrides.clear()

    def _agra_forward_crossover_train(self) -> RuntimeTrain | None:
        crossover_id = "XOVER-AGRA-01"
        crossover = self._crossovers.get(crossover_id)
        if crossover is None:
            return None
        for train in self.trains:
            if self.sim_time_s < train.departure_time_s:
                continue
            plan = self._train_plan_for_crossover(train, crossover_id)
            if plan is None or plan.reverse_after_change:
                continue
            if crossover_id not in train.completed_crossovers:
                continue
            if train.current_track_id != crossover.to_track_id or train.sign <= 0:
                continue
            return train
        return None

    def _agra_dn07_has_other_occupancy(
        self,
        forward_crossover_train: RuntimeTrain,
        exclude: RuntimeTrain | None,
    ) -> bool:
        block_index = self.config.route.block_index("BLK-07")
        block = self.config.route.blocks[block_index]
        block_start = self.config.route.block_start_distance_m(block.block_id)
        block_end = block_start + block.length_m
        for train in self.trains:
            if train is forward_crossover_train or train is exclude:
                continue
            if train.completed or self.sim_time_s < train.departure_time_s:
                continue
            if "TRACK-DOWN" not in self._occupancy_tracks(train):
                continue
            body_start, body_end = self._body_bounds(train)
            if self._intervals_overlap(body_start, body_end, block_start, block_end):
                return True
        return False

    def _agra_turnaround_has_reversed(self) -> bool:
        crossover_id = "XOVER-AGRA-01"
        crossover = self._crossovers.get(crossover_id)
        if crossover is None:
            return False
        for train in self.trains:
            plan = self._train_plan_for_crossover(train, crossover_id)
            if plan is None or not plan.reverse_after_change:
                continue
            if crossover_id not in train.completed_crossovers:
                continue
            if train.current_track_id != crossover.to_track_id:
                continue
            if train.sign < 0 and getattr(train, "pending_turnaround_crossover_id", None) is None:
                return True
        return False

    def signal_aspect(self, signal, exclude: RuntimeTrain | None = None) -> SignalAspect:
        override = self.manual_signal_overrides().get(signal.signal_id)
        if override is not None:
            return override

        forward_crossover_train = self._agra_forward_crossover_train()
        if forward_crossover_train is not None:
            if signal.signal_id == "DN-08":
                return SignalAspect.RED
            if signal.signal_id == "DN-07":
                if self._agra_turnaround_has_reversed():
                    return super().signal_aspect(signal, exclude=exclude)
                if self._agra_dn07_has_other_occupancy(forward_crossover_train, exclude):
                    return SignalAspect.RED
                return SignalAspect.YELLOW

        return super().signal_aspect(signal, exclude=exclude)

    def _directional_signals(self, train: RuntimeTrain):
        return [
            signal
            for signal in self.config.signals
            if signal.track_id == train.current_track_id and signal.direction == train.direction
        ]

    def _restrictive_approach_signal(self, train: RuntimeTrain):
        signals = self._directional_signals(train)
        ahead = []
        behind = []
        for signal in signals:
            position = self._signal_position(signal)
            signed_distance = (position - train.route_position_m) * train.sign
            if signed_distance >= -1e-6:
                ahead.append((signed_distance, signal))
            elif signal.signal_type == SignalType.STANDARD:
                behind.append((-signed_distance, signal))

        ahead.sort(key=lambda item: item[0])
        behind.sort(key=lambda item: item[0])
        next_signal = next(
            (
                signal
                for _, signal in ahead
                if signal.signal_type == SignalType.STANDARD
                or self.signal_aspect(signal, exclude=train) != SignalAspect.GREEN
            ),
            None,
        )
        previous_signal = behind[0][1] if behind else None

        if next_signal is not None:
            next_aspect = self.signal_aspect(next_signal, exclude=train)
            if next_aspect == SignalAspect.YELLOW:
                return next_signal

            if previous_signal is not None:
                previous_aspect = self.signal_aspect(previous_signal, exclude=train)
                if previous_aspect == SignalAspect.YELLOW and next_aspect in {SignalAspect.YELLOW, SignalAspect.RED}:
                    return previous_signal

        return None

    def _platform_for_station(self, station_id: str, track_id: str):
        station = next((item for item in self.config.stations if item.station_id == station_id), None)
        if station is None:
            return None
        return next((platform for platform in station.platforms if platform.track_id == track_id), None)

    def _platform_bounds(self, platform) -> tuple[float, float]:
        center = self.config.route.block_start_distance_m(platform.block_id) + platform.position_in_block_m
        half = platform.length_m / 2.0
        return center - half, center + half

    def _platform_occupant(self, train: RuntimeTrain, station_id: str):
        platform = self._platform_for_station(station_id, train.current_track_id)
        if platform is None:
            return None
        platform_start, platform_end = self._platform_bounds(platform)
        for other in self.trains:
            if other is train or other.completed or self.sim_time_s < other.departure_time_s:
                continue
            if train.current_track_id not in self._occupancy_tracks(other):
                continue
            body_start, body_end = self._body_bounds(other)
            if self._intervals_overlap(body_start, body_end, platform_start, platform_end):
                return other
        return None

    def _station_occupancy_targets(self, train: RuntimeTrain) -> list[Target]:
        targets: list[Target] = []
        for stop in train.station_stops:
            if stop.station_id in train.served_stations:
                continue
            platform = self._platform_for_station(stop.station_id, train.current_track_id)
            if platform is None:
                continue
            occupant = self._platform_occupant(train, stop.station_id)
            if occupant is None:
                continue

            platform_start, platform_end = self._platform_bounds(platform)
            entrance = platform_start if train.sign > 0 else platform_end
            stop_position = entrance - train.sign * self.STATION_ENTRY_MARGIN_M
            if self._ahead(train, stop_position) is not None:
                targets.append(Target(
                    stop_position,
                    0.0,
                    f"STATION_OCCUPIED:{stop.station_id}:{occupant.train.train_id}",
                    True,
                ))
        return targets

    def _targets(self, train: RuntimeTrain):
        targets = [
            target
            for target in super()._targets(train)
            if not target.reason.startswith("STATION:")
        ]

        targets.extend(self._station_occupancy_targets(train))

        for stop in train.station_stops:
            if stop.station_id in train.served_stations:
                continue
            platform_center = self._platform_position(stop.station_id, train.current_track_id)
            if platform_center is None:
                continue
            stop_position = platform_center + train.sign * (train.train.length_m / 2.0)
            if self._ahead(train, stop_position) is not None:
                targets.append(Target(
                    stop_position,
                    0.0,
                    f"STATION:{stop.station_id}",
                    True,
                    stop.station_id,
                ))

        if not any(plan.reverse_after_change for plan in train.track_changes):
            return targets

        next_stop = next(
            (stop for stop in train.station_stops if stop.station_id not in train.served_stations),
            None,
        )
        if next_stop is None:
            return targets

        next_station_reason = f"STATION:{next_stop.station_id}"
        return [
            target
            for target in targets
            if not target.reason.startswith("STATION:") or target.reason == next_station_reason
        ]

    def _desired_speed(
        self,
        train: RuntimeTrain,
        ceiling: float,
        weather: WeatherCondition,
    ) -> tuple[float, str]:
        desired, reason = super()._desired_speed(train, ceiling, weather)
        signal = self._restrictive_approach_signal(train)
        if signal is not None and desired > self.YELLOW_APPROACH_SPEED_KMH:
            return self.YELLOW_APPROACH_SPEED_KMH, f"YELLOW_APPROACH:{signal.signal_id}"
        return desired, reason
