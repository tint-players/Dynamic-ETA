from __future__ import annotations

from simulator.models import SimulationConfig, TrainDirection


def _timeline(entries):
    return [entry.model_dump(mode="json") for entry in entries]


def _endpoint_m(config: SimulationConfig, endpoint) -> float:
    return config.route.block_start_distance_m(endpoint.block_id) + endpoint.position_in_block_m


def config_for_visualization(config: SimulationConfig) -> dict:
    route = config.route
    blocks = []
    for block in route.blocks:
        start_m = route.block_start_distance_m(block.block_id)
        blocks.append({**block.model_dump(mode="json"), "route_start_m": start_m, "route_end_m": start_m + block.length_m})

    signals = []
    for signal in config.signals:
        block_start = route.block_start_distance_m(signal.protected_block_id)
        block = route.blocks[route.block_index(signal.protected_block_id)]
        if signal.position_in_block_m is not None:
            position = block_start + signal.position_in_block_m
        else:
            position = block_start if signal.direction == TrainDirection.FORWARD else block_start + block.length_m
        signals.append({**signal.model_dump(mode="json"), "route_position_m": position})

    def restrictions(items):
        result = []
        for restriction in items:
            block_start = route.block_start_distance_m(restriction.block_id)
            result.append({**restriction.model_dump(mode="json"), "route_start_m": block_start + restriction.start_position_m, "route_end_m": block_start + restriction.end_position_m})
        return result

    crossings = []
    for crossing in config.environment.crossings:
        block_start = route.block_start_distance_m(crossing.block_id)
        crossings.append({**crossing.model_dump(mode="json"), "route_position_m": block_start + crossing.position_in_block_m})

    crossovers = []
    for crossover in config.crossovers:
        block_start = route.block_start_distance_m(crossover.block_id)
        crossovers.append({
            **crossover.model_dump(mode="json"),
            "route_start_m": block_start + crossover.start_position_m,
            "route_end_m": block_start + crossover.end_position_m,
            "route_mid_m": block_start + (crossover.start_position_m + crossover.end_position_m) / 2,
        })

    stations = []
    for station in config.stations:
        platforms = []
        for platform in station.platforms:
            route_position = route.block_start_distance_m(platform.block_id) + platform.position_in_block_m
            platforms.append({**platform.model_dump(mode="json"), "route_position_m": route_position})
        stations.append({"station_id": station.station_id, "station_name": station.station_name, "platforms": platforms})

    trains = [{
        "train": config.train.model_dump(mode="json"), "journey": config.journey.model_dump(mode="json"),
        "departure_time_s": 0.0, "station_stops": [x.model_dump(mode="json") for x in config.primary_station_stops],
        "track_changes": [x.model_dump(mode="json") for x in config.primary_track_changes],
        "source_route_m": _endpoint_m(config, config.journey.source), "destination_route_m": _endpoint_m(config, config.journey.destination),
    }]
    for run in config.additional_train_runs:
        trains.append({
            "train": run.train.model_dump(mode="json"), "journey": run.journey.model_dump(mode="json"),
            "departure_time_s": run.departure_time_s, "station_stops": [x.model_dump(mode="json") for x in run.station_stops],
            "track_changes": [x.model_dump(mode="json") for x in run.track_changes],
            "source_route_m": _endpoint_m(config, run.journey.source), "destination_route_m": _endpoint_m(config, run.journey.destination),
        })

    source_m = _endpoint_m(config, config.journey.source)
    destination_m = _endpoint_m(config, config.journey.destination)
    return {
        "route": {"route_id": route.route_id, "route_name": route.route_name, "track_ids": route.track_ids, "total_length_m": route.total_length_m, "blocks": blocks},
        "signals": signals, "dynamic_signalling": config.dynamic_signalling, "stations": stations, "crossovers": crossovers, "trains": trains,
        "train": config.train.model_dump(mode="json"),
        "journey": {**config.journey.model_dump(mode="json"), "source_route_m": source_m, "destination_route_m": destination_m},
        "environment": {
            "weather": [{"block_id": x.block_id, "timeline": _timeline(x.timeline)} for x in config.environment.weather],
            "signal_states": [{"signal_id": x.signal_id, "timeline": _timeline(x.timeline)} for x in config.environment.signal_states],
            "temporary_speed_restrictions": restrictions(config.environment.temporary_speed_restrictions),
            "maintenance_restrictions": restrictions(config.environment.maintenance_restrictions), "crossings": crossings,
        },
        "simulation": config.simulation.model_dump(mode="json"),
    }
