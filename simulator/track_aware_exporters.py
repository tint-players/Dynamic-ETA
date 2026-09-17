from __future__ import annotations

import pandas as pd

from .exporters import BatchExporter, BlockVisitExporter, ParquetTelemetryExporter
from .models import TelemetryFrame
from .track_blocks import track_block_id


class TrackAwareBatchExporter(BatchExporter):
    """Raw multi-scenario exporter that preserves canonical track-block identity."""

    def add(self, frames: list[TelemetryFrame]) -> None:
        for frame in frames:
            row = frame.model_dump(mode="json")
            row["track_block_id"] = track_block_id(frame.track_id, frame.current_block_id)
            self._rows.append(row)


class TrackAwareParquetTelemetryExporter(ParquetTelemetryExporter):
    """Canonical Phase-2 tick exporter for track-specific operational state.

    Shared logical block geometry remains available in ``block_id`` and the
    existing block geometry columns. ``track_block_id`` is the operational
    identity consumed by downstream graph/dataset code.
    """

    @classmethod
    def _restriction_is_active(cls, restriction, frame: TelemetryFrame) -> bool:
        if restriction.track_id is not None and restriction.track_id != frame.track_id:
            return False
        return super()._restriction_is_active(restriction, frame)

    def to_dataframe(self, frames: list[TelemetryFrame]) -> pd.DataFrame:
        dataframe = super().to_dataframe(frames)
        if dataframe.empty:
            return dataframe
        dataframe.insert(
            dataframe.columns.get_loc("block_id") + 1,
            "track_block_id",
            [
                track_block_id(str(track_id), str(block_id))
                for track_id, block_id in zip(dataframe["track_id"], dataframe["block_id"])
            ],
        )
        return dataframe


class TrackBlockVisitExporter(BlockVisitExporter):
    """Post-run outcome exporter with one visit per physical track-block section.

    The legacy exporter segments by logical block and direction. For a crossover
    that changes tracks while the train remains inside the same logical block,
    that merges two operational sections. This adapter reuses the established
    outcome calculations while segmenting on ``track_block_id`` instead.
    """

    def to_dataframe(self, telemetry: pd.DataFrame) -> pd.DataFrame:
        if telemetry.empty:
            return pd.DataFrame()

        work = telemetry.copy()
        if "track_block_id" not in work.columns:
            work["track_block_id"] = [
                track_block_id(str(track_id), str(block_id))
                for track_id, block_id in zip(work["track_id"], work["block_id"])
            ]

        logical_block_by_track_block = (
            work[["track_block_id", "block_id"]]
            .drop_duplicates()
            .set_index("track_block_id")["block_id"]
            .to_dict()
        )

        # Reuse the mature visit/outcome aggregation by temporarily making the
        # canonical operational identity the segmentation key.
        work["block_id"] = work["track_block_id"]
        visits = super().to_dataframe(work)
        if visits.empty:
            return visits

        visits = visits.rename(columns={"block_id": "track_block_id"})
        visits.insert(
            visits.columns.get_loc("track_block_id") + 1,
            "block_id",
            visits["track_block_id"].map(logical_block_by_track_block),
        )
        return visits
