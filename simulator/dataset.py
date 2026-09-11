from __future__ import annotations

from collections import defaultdict

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


def label_completed_multi_train_journey(frames: list[TelemetryFrame]) -> list[TelemetryFrame]:
    """Label each train independently after the complete network run.

    Pre-departure rows are retained because they are observable simulation state;
    their remaining-time label is measured to that train's own arrival.
    """
    if not frames:
        return []
    grouped: dict[str, list[TelemetryFrame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.train_id].append(frame)

    labels: dict[tuple[str, int, float], TelemetryFrame] = {}
    for train_id, train_frames in grouped.items():
        completed = [frame for frame in train_frames if frame.completed or frame.distance_to_destination_m <= 1e-6]
        if not completed:
            raise ValueError(f"Cannot label incomplete journey for train {train_id}")
        arrival = min(frame.sim_time_s for frame in completed)
        first_time = train_frames[0].sim_time_s
        total = arrival - first_time
        for frame in train_frames:
            labels[(train_id, frame.tick, frame.sim_time_s)] = frame.model_copy(update={
                "actual_remaining_time_s": max(0.0, arrival - frame.sim_time_s),
                "actual_arrival_simulation_s": arrival,
                "total_journey_time_s": total,
            })

    return [labels[(frame.train_id, frame.tick, frame.sim_time_s)] for frame in frames]
