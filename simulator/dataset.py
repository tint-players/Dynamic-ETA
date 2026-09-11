from __future__ import annotations

from .models import TelemetryFrame


def label_completed_journey(frames: list[TelemetryFrame]) -> list[TelemetryFrame]:
    if not frames:
        return []
    arrival = frames[-1].sim_time_s
    if frames[-1].distance_to_destination_m > 1e-6 or frames[-1].speed_kmh > 1e-6:
        raise ValueError("Cannot label an incomplete journey")
    total = arrival - frames[0].sim_time_s
    return [
        frame.model_copy(update={
            "actual_remaining_time_s": max(0.0, arrival - frame.sim_time_s),
            "actual_arrival_simulation_s": arrival,
            "total_journey_time_s": total,
        })
        for frame in frames
    ]
