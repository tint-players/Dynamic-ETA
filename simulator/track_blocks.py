from __future__ import annotations

from dataclasses import dataclass

from .models import Route, TrackBlock


@dataclass(frozen=True, slots=True)
class TrackBlockIdentity:
    """Canonical operational identity for one physical track through one logical block.

    Route.blocks remains the shared corridor geometry. A TrackBlockIdentity is the
    operational section used for occupancy, restrictions, signals, telemetry and
    later graph nodes. Weather can remain attached to the shared logical block.
    """

    track_block_id: str
    track_id: str
    block_id: str
    block_index: int
    track_index: int
    route_start_m: float
    route_end_m: float
    geometry: TrackBlock

    @property
    def length_m(self) -> float:
        return self.geometry.length_m


def track_block_id(track_id: str, block_id: str) -> str:
    """Return the stable public identifier for a track-specific block section."""

    if not track_id:
        raise ValueError("track_id must not be empty")
    if not block_id:
        raise ValueError("block_id must not be empty")

    if track_id.startswith("TRACK-"):
        track_label = track_id.removeprefix("TRACK-")
    else:
        track_label = track_id
    return f"{track_label}-{block_id}"


def enumerate_track_blocks(route: Route) -> list[TrackBlockIdentity]:
    """Expand shared block geometry into deterministic track-specific sections.

    Ordering is track-major and follows route.track_ids and route.blocks exactly.
    For the Delhi-Agra route this produces UP-BLK-01..08 followed by
    DOWN-BLK-01..08, matching the intended ML graph node ordering.
    """

    sections: list[TrackBlockIdentity] = []
    starts: dict[str, float] = {}
    running = 0.0
    for block in route.blocks:
        starts[block.block_id] = running
        running += block.length_m

    seen_ids: set[str] = set()
    for track_index, track_id in enumerate(route.track_ids):
        for block_index, block in enumerate(route.blocks):
            section_id = track_block_id(track_id, block.block_id)
            if section_id in seen_ids:
                raise ValueError(f"Duplicate track_block_id generated: {section_id}")
            seen_ids.add(section_id)
            route_start_m = starts[block.block_id]
            sections.append(
                TrackBlockIdentity(
                    track_block_id=section_id,
                    track_id=track_id,
                    block_id=block.block_id,
                    block_index=block_index,
                    track_index=track_index,
                    route_start_m=route_start_m,
                    route_end_m=route_start_m + block.length_m,
                    geometry=block,
                )
            )
    return sections


def get_track_block(route: Route, track_id: str, block_id: str) -> TrackBlockIdentity:
    """Resolve one operational section and reject unknown track/block references."""

    if track_id not in route.track_ids:
        raise KeyError(f"Unknown track_id: {track_id}")
    block_index = route.block_index(block_id)
    block = route.blocks[block_index]
    track_index = route.track_ids.index(track_id)
    route_start_m = route.block_start_distance_m(block_id)
    return TrackBlockIdentity(
        track_block_id=track_block_id(track_id, block_id),
        track_id=track_id,
        block_id=block_id,
        block_index=block_index,
        track_index=track_index,
        route_start_m=route_start_m,
        route_end_m=route_start_m + block.length_m,
        geometry=block,
    )


def locate_track_block(route: Route, track_id: str, route_position_m: float) -> tuple[TrackBlockIdentity, float]:
    """Locate a route position on a specific physical track."""

    block_index, block, position_in_block_m = route.locate(route_position_m)
    section = get_track_block(route, track_id, block.block_id)
    if section.block_index != block_index:
        raise RuntimeError("track-block lookup disagrees with route location")
    return section, position_in_block_m
