from __future__ import annotations

from .models import SimulationConfig
from .track_blocks import get_track_block


def validate_track_block_config(config: SimulationConfig) -> SimulationConfig:
    """Validate operational objects against canonical track-block identity.

    Shared block geometry and weather intentionally remain block-scoped. On a
    multi-track route, operational restrictions must identify exactly one track.
    Signals and platforms already carry an explicit track_id and are resolved here
    against the same canonical track-block namespace.

    Level crossings are physical cross-track infrastructure: one configured
    crossing applies to all route tracks unless a future crossing model introduces
    an explicit subset. This is intentional and not treated as an ambiguous
    block-only restriction.
    """

    multi_track = len(config.route.track_ids) > 1

    for restriction in [
        *config.environment.temporary_speed_restrictions,
        *config.environment.maintenance_restrictions,
    ]:
        if multi_track and restriction.track_id is None:
            raise ValueError(
                f"Restriction {restriction.restriction_id} must specify track_id on a multi-track route"
            )
        if restriction.track_id is not None:
            get_track_block(config.route, restriction.track_id, restriction.block_id)

    for signal in config.signals:
        get_track_block(config.route, signal.track_id, signal.protected_block_id)

    for station in config.stations:
        for platform in station.platforms:
            get_track_block(config.route, platform.track_id, platform.block_id)

    for crossover in config.crossovers:
        get_track_block(config.route, crossover.from_track_id, crossover.block_id)
        get_track_block(config.route, crossover.to_track_id, crossover.block_id)

    return config
