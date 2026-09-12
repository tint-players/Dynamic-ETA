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
    restrictive, and preserves configured station-stop order for turnaround
    trains so an itinerary can continue naturally onto the return leg.
    """

    YELLOW_APPROACH_SPEED_KMH = 60.0

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

    def _targets(self, train: RuntimeTrain):
        targets = super()._targets(train)

        centered_targets = []
        for target in targets:
            if target.station_id is not None and target.reason.startswith("STATION:"):
                centered_targets.append(Target(
                    target.position_m + train.sign * (train.train.length_m / 2.0),
                    target.speed_kmh,
                    target.reason,
                    target.hard_stop,
                    target.station_id,
                ))
            else:
                centered_targets.append(target)
        targets = centered_targets

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
