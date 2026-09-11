from __future__ import annotations

from simulator.models import SimulationConfig


def _timeline(entries):
    return [entry.model_dump(mode="json") for entry in entries]


def config_for_visualization(config: SimulationConfig) -> dict:
    """Serialize only backend-owned configuration needed to draw the route.

    Route positions are derived once from the canonical Python config. The
    frontend may scale these values into pixels, but never recomputes physics.
    """
    route = config.route
    blocks = []
    for block in route.blocks:
        start_m = route.block_start_distance_m(block.block_id)
        blocks.append(
            {
                **block.model_dump(mode="json"),
                "route_start_m": start_m,
                "route_end_m": start_m + block.length_m,
            }
        )

    signals = []
    for signal in config.signals:
        signals.append(
            {
                **signal.model_dump(mode="json"),
                "route_position_m": route.block_start_distance_m(signal.protected_block_id),
            }
        )

    tsrs = []
    for restriction in config.environment.temporary_speed_restrictions:
        block_start = route.block_start_distance_m(restriction.block_id)
        tsrs.append(
            {
                **restriction.model_dump(mode="json"),
                "route_start_m": block_start + restriction.start_position_m,
                "route_end_m": block_start + restriction.end_position_m,
            }
        )

    maintenance = []
    for restriction in config.environment.maintenance_restrictions:
        block_start = route.block_start_distance_m(restriction.block_id)
        maintenance.append(
            {
                **restriction.model_dump(mode="json"),
                "route_start_m": block_start + restriction.start_position_m,
                "route_end_m": block_start + restriction.end_position_m,
            }
        )

    crossings = []
    for crossing in config.environment.crossings:
        block_start = route.block_start_distance_m(crossing.block_id)
        crossings.append(
            {
                "crossing_id": crossing.crossing_id,
                "block_id": crossing.block_id,
                "position_in_block_m": crossing.position_in_block_m,
                "route_position_m": block_start + crossing.position_in_block_m,
                "timeline": _timeline(crossing.timeline),
            }
        )

    weather = [
        {
            "block_id": schedule.block_id,
            "timeline": _timeline(schedule.timeline),
        }
        for schedule in config.environment.weather
    ]
    signal_states = [
        {
            "signal_id": schedule.signal_id,
            "timeline": _timeline(schedule.timeline),
        }
        for schedule in config.environment.signal_states
    ]

    source_m = route.block_start_distance_m(config.journey.source.block_id) + config.journey.source.position_in_block_m
    destination_m = (
        route.block_start_distance_m(config.journey.destination.block_id)
        + config.journey.destination.position_in_block_m
    )

    return {
        "route": {
            "route_id": route.route_id,
            "route_name": route.route_name,
            "total_length_m": route.total_length_m,
            "blocks": blocks,
        },
        "signals": signals,
        "train": config.train.model_dump(mode="json"),
        "journey": {
            **config.journey.model_dump(mode="json"),
            "source_route_m": source_m,
            "destination_route_m": destination_m,
        },
        "environment": {
            "weather": weather,
            "signal_states": signal_states,
            "temporary_speed_restrictions": tsrs,
            "maintenance_restrictions": maintenance,
            "crossings": crossings,
        },
        "simulation": config.simulation.model_dump(mode="json"),
    }
