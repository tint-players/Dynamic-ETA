from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .ml_features import MLFeatureBuilder
from .ml_graph import MLGraphTopologyBuilder
from .models import SimulationConfig, TelemetryFrame


@dataclass(frozen=True, slots=True)
class MLModelInputs:
    """Inference-safe model inputs for one train at one simulation second."""

    x_seq: tuple[tuple[float, ...], ...]
    x_graph: tuple[tuple[float, ...], ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    operational_edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    operational_edge_types: tuple[str, ...]
    current_node_index: int
    route_mask: tuple[float, ...]
    x_context: tuple[float, ...]
    node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MLTrainingSample:
    """One labelled one-second training example plus run provenance."""

    inputs: MLModelInputs
    y: float
    run_id: str
    scenario_id: str
    random_seed: int
    train_id: str
    sim_time_s: float


class MLSampleBuilder:
    """Combine shared features and static graph topology into model samples.

    ``build_inputs`` is safe for live inference and never reads post-run labels.
    ``build_training_sample`` is an offline-only wrapper that attaches the
    completed-journey remaining-time target and provenance after the exact same
    input tensors have already been built.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.feature_builder = MLFeatureBuilder(config)
        self.graph_builder = MLGraphTopologyBuilder(config)
        self.topology = self.graph_builder.build()

        if self.feature_builder.node_ids != self.topology.node_ids:
            raise RuntimeError("ML feature and topology node orderings disagree")

    @staticmethod
    def _current_target(
        current_frames: Iterable[TelemetryFrame],
        target_train_id: str,
    ) -> tuple[list[TelemetryFrame], TelemetryFrame]:
        current = list(current_frames)
        target = next((frame for frame in current if frame.train_id == target_train_id), None)
        if target is None:
            raise ValueError(f"No current frame for target train {target_train_id}")
        if any(abs(frame.sim_time_s - target.sim_time_s) > 1e-6 for frame in current):
            raise ValueError("current_frames must all describe the same simulation second")
        return current, target

    def build_inputs(
        self,
        *,
        history: Iterable[TelemetryFrame],
        current_frames: Iterable[TelemetryFrame],
        target_train_id: str,
    ) -> MLModelInputs:
        current, _ = self._current_target(current_frames, target_train_id)
        features = self.feature_builder.build(
            history=history,
            current_frames=current,
            target_train_id=target_train_id,
        )
        if features.node_ids != self.topology.node_ids:
            raise RuntimeError("ML feature and topology node orderings disagree")

        return MLModelInputs(
            x_seq=features.sequence,
            x_graph=features.graph,
            edge_index=self.topology.edge_index,
            operational_edge_index=self.topology.operational_edge_index,
            operational_edge_types=self.topology.operational_edge_types,
            current_node_index=features.current_node_index,
            route_mask=self.graph_builder.route_mask(target_train_id),
            x_context=features.context,
            node_ids=features.node_ids,
        )

    def build_training_sample(
        self,
        *,
        history: Iterable[TelemetryFrame],
        current_frames: Iterable[TelemetryFrame],
        target_train_id: str,
    ) -> MLTrainingSample:
        current, target = self._current_target(current_frames, target_train_id)

        if not target.run_id:
            raise ValueError("Training sample target is missing run_id")
        if target.random_seed is None:
            raise ValueError(f"Training sample {target.run_id} is missing random_seed")
        if target.actual_remaining_time_s is None:
            raise ValueError(f"Training sample {target.run_id} is missing actual_remaining_time_s")
        if target.actual_remaining_time_s < 0:
            raise ValueError(f"Training sample {target.run_id} has negative actual_remaining_time_s")

        inputs = self.build_inputs(
            history=history,
            current_frames=current,
            target_train_id=target_train_id,
        )
        return MLTrainingSample(
            inputs=inputs,
            y=float(target.actual_remaining_time_s),
            run_id=target.run_id,
            scenario_id=target.scenario_id,
            random_seed=target.random_seed,
            train_id=target.train_id,
            sim_time_s=float(target.sim_time_s),
        )

    def build_inputs_from_records(
        self,
        *,
        history_records: Iterable[Mapping[str, object]],
        current_records: Iterable[Mapping[str, object]],
        target_train_id: str,
    ) -> MLModelInputs:
        history = [TelemetryFrame.model_validate(dict(record)) for record in history_records]
        current = [TelemetryFrame.model_validate(dict(record)) for record in current_records]
        return self.build_inputs(
            history=history,
            current_frames=current,
            target_train_id=target_train_id,
        )

    def build_training_sample_from_records(
        self,
        *,
        history_records: Iterable[Mapping[str, object]],
        current_records: Iterable[Mapping[str, object]],
        target_train_id: str,
    ) -> MLTrainingSample:
        history = [TelemetryFrame.model_validate(dict(record)) for record in history_records]
        current = [TelemetryFrame.model_validate(dict(record)) for record in current_records]
        return self.build_training_sample(
            history=history,
            current_frames=current,
            target_train_id=target_train_id,
        )
