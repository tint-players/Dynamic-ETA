"""
Output adapters.

Two independent exporters sit on top of the same TelemetryFrame objects:

  - BatchExporter: accumulates frames and writes them to Parquet/CSV in one
    shot — efficient for bulk training-data generation, no JSON involved.

  - LiveExporter: serializes a single frame to a JSON string, ready to push
    over a WebSocket / Redis Pub/Sub channel to a live consumer.

Keeping these separate means adding a third format later (e.g. Arrow feather
files, or a Kafka producer) doesn't touch the engine at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models import TelemetryFrame


class BatchExporter:
    def __init__(self):
        self._rows: list[dict] = []

    def add(self, frames: list[TelemetryFrame]) -> None:
        for f in frames:
            self._rows.append(f.model_dump(mode="json"))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def to_parquet(self, path: str | Path) -> None:
        self.to_dataframe().to_parquet(path, index=False)

    def to_csv(self, path: str | Path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def clear(self) -> None:
        self._rows = []


class LiveExporter:
    @staticmethod
    def to_json(frame: TelemetryFrame) -> str:
        return frame.model_dump_json()

    @staticmethod
    def to_json_batch(frames: list[TelemetryFrame]) -> str:
        return json.dumps([f.model_dump(mode="json") for f in frames], default=str)
