from __future__ import annotations

from .models import SignalAspect, WeatherCondition
from .network_engine import RuntimeTrain
from .network_engine_v4 import NetworkSimulationEngineV4


class NetworkSimulationEngineV4Restrictive(NetworkSimulationEngineV4):
    """V4 network engine with a conservative three-aspect approach-speed rule.

    The existing v4 RED stop targets, crossing protection, separation and
    crossover interlocking remain unchanged. This layer only prevents a train
    from accelerating back toward line speed while the governing signal is
    YELLOW, and keeps that caution speed after passing a YELLOW while the next
    signal remains restrictive.
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
