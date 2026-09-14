from __future__ import annotations

from dataclasses import dataclass

from .models import SimulationConfig, TrainDirection
from .track_blocks import enumerate_track_blocks, track_block_id


ROUTE_SUCCESSOR_EDGE = "ROUTE_SUCCESSOR"
CROSSOVER_EDGE = "CROSSOVER"


@dataclass(frozen=True, slots=True)
class MLGraphTopology:
    """Static railway topology aligned to canonical track-block node ordering.

    ``operational_edge_index`` preserves the real directed railway movement
    semantics. ``edge_index`` is the symmetric message-passing view used by a
    GraphSAGE-style encoder so information can flow in both directions while
    the directed truth remains available separately.
    """

    node_ids: tuple[str, ...]
    operational_edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    operational_edge_types: tuple[str, ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_operational_edges(self) -> int:
        return len(self.operational_edge_types)

    @property
    def num_message_edges(self) -> int:
        return len(self.edge_index[0])


class MLGraphTopologyBuilder:
    """Build the static track-block graph consumed by ETA models."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._sections = enumerate_track_blocks(config.route)
        self.node_ids = tuple(section.track_block_id for section in self._sections)
        self._node_index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        self._train_runs = {config.train.train_id: (config.train, config.journey, config.primary_track_changes)}
        self._train_runs.update(
            {
                run.train.train_id: (run.train, run.journey, run.track_changes)
                for run in config.additional_train_runs
            }
        )

    def _track_direction(self, track_id: str) -> TrainDirection:
        signal_directions = {
            signal.direction
            for signal in self.config.signals
            if signal.track_id == track_id
        }
        if len(signal_directions) == 1:
            return next(iter(signal_directions))
        if len(signal_directions) > 1:
            raise ValueError(f"Track {track_id} has ambiguous signal directions for ML topology")

        token = track_id.upper()
        if "DOWN" in token or "REVERSE" in token:
            return TrainDirection.REVERSE
        if "UP" in token or "FORWARD" in token:
            return TrainDirection.FORWARD
        raise ValueError(
            f"Cannot infer operational direction for track {track_id}; add directional signals"
        )

    def _node(self, track_id: str, block_id: str) -> int:
        node_id = track_block_id(track_id, block_id)
        try:
            return self._node_index[node_id]
        except KeyError as exc:
            raise ValueError(f"Unknown canonical track-block {node_id}") from exc

    def build(self) -> MLGraphTopology:
        directed_edges: list[tuple[int, int, str]] = []
        block_ids = [block.block_id for block in self.config.route.blocks]

        for track_id in self.config.route.track_ids:
            direction = self._track_direction(track_id)
            ordered_blocks = block_ids if direction == TrainDirection.FORWARD else list(reversed(block_ids))
            for current_block, next_block in zip(ordered_blocks, ordered_blocks[1:]):
                directed_edges.append(
                    (
                        self._node(track_id, current_block),
                        self._node(track_id, next_block),
                        ROUTE_SUCCESSOR_EDGE,
                    )
                )

        for crossover in self.config.crossovers:
            directed_edges.append(
                (
                    self._node(crossover.from_track_id, crossover.block_id),
                    self._node(crossover.to_track_id, crossover.block_id),
                    CROSSOVER_EDGE,
                )
            )

        # Keep the operational list stable and de-duplicated. This matters for
        # reproducible graph tensors and model checkpoints.
        unique_directed: list[tuple[int, int, str]] = []
        seen_directed: set[tuple[int, int, str]] = set()
        for edge in directed_edges:
            if edge not in seen_directed:
                seen_directed.add(edge)
                unique_directed.append(edge)

        op_src = tuple(edge[0] for edge in unique_directed)
        op_dst = tuple(edge[1] for edge in unique_directed)
        edge_types = tuple(edge[2] for edge in unique_directed)

        # GraphSAGE aggregates neighbours rather than modelling railway travel
        # direction itself. Supply both orientations for message passing, while
        # keeping operational_edge_index as the authoritative directed graph.
        message_edges: list[tuple[int, int]] = []
        seen_message: set[tuple[int, int]] = set()
        for src, dst, _ in unique_directed:
            for pair in ((src, dst), (dst, src)):
                if pair not in seen_message:
                    seen_message.add(pair)
                    message_edges.append(pair)

        return MLGraphTopology(
            node_ids=self.node_ids,
            operational_edge_index=(op_src, op_dst),
            operational_edge_types=edge_types,
            edge_index=(
                tuple(edge[0] for edge in message_edges),
                tuple(edge[1] for edge in message_edges),
            ),
        )

    def route_mask(self, train_id: str) -> tuple[float, ...]:
        """Return a static configured-route envelope for one train.

        The mask covers the journey's source-to-destination block span on the
        train's starting track and also marks both endpoints of every explicitly
        configured crossover. It is intentionally configuration-derived rather
        than inferred from future runtime outcomes, so it is safe for live use.
        """
        try:
            train, journey, track_changes = self._train_runs[train_id]
        except KeyError as exc:
            raise KeyError(f"Unknown train_id for ML route mask: {train_id}") from exc

        source_index = self.config.route.block_index(journey.source.block_id)
        destination_index = self.config.route.block_index(journey.destination.block_id)
        low, high = sorted((source_index, destination_index))
        included = {
            track_block_id(train.track_id, self.config.route.blocks[index].block_id)
            for index in range(low, high + 1)
        }

        crossover_by_id = {item.crossover_id: item for item in self.config.crossovers}
        for change in track_changes:
            crossover = crossover_by_id[change.crossover_id]
            included.add(track_block_id(crossover.from_track_id, crossover.block_id))
            included.add(track_block_id(crossover.to_track_id, crossover.block_id))

        return tuple(1.0 if node_id in included else 0.0 for node_id in self.node_ids)
