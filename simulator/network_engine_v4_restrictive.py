from __future__ import annotations

from .models import SignalAspect, WeatherCondition
from .network_engine import RuntimeTrain, Target
from .network_engine_v4 import NetworkSimulationEngineV4


class NetworkSimulationEngineV4Restrictive(NetworkSimulationEngineV4):
    """V4 network engine with conservative restrictive-signal approach control.

    The existing v4 RED stop targets, crossing protection, separation and
    crossover interlocking remain unchanged. This layer prevents a train from
    accelerating back toward line speed while the governing signal is YELLOW,
    keeps that caution speed after passing a YELLOW while the next signal remains
    restrictive, preserves configured station-stop order for turnaround trains,
    and adds a conservative station-entry hold when the required platform is
    physically occupied by another active train.
    """

    YELLOW_APPROACH_SPEED_KMH = 60.0
    STATION_ENTRY_MARGIN_M = 20.0

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
            else:
                behind.append((-signed_distance, signal))

        ahead.sort(key=lambda item: item[0])
        behind.sort(key=lambda item: item[0])
        next_signal = ahead[0][1] if ahead else None
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
        """Hold before a required platform only when another train occupies it.

        Existing RED-signal, separation and crossover targets remain in the target
        set and therefore naturally win when they provide an earlier safe stop.
        This target is a fallback for station layouts without a suitably placed
        protecting signal.
        """
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
        # Build all non-station targets exactly as V4 already does, then rebuild
        # station targets around the train-body centre. We cannot simply shift the
        # station targets returned by super()._targets(): once the train front has
        # passed the platform centre, V4 would stop returning that original target
        # even though the shifted front target is still ahead.
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
