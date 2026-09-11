from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .models import TelemetryFrame


class BatchExporter:
    def __init__(self):
        self._rows: list[dict] = []

    def add(self, frames: list[TelemetryFrame]) -> None:
        self._rows.extend(frame.model_dump(mode="json") for frame in frames)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def to_parquet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    def clear(self) -> None:
        self._rows = []


class LiveExporter:
    @staticmethod
    def to_json(frame: TelemetryFrame) -> str:
        return frame.model_dump_json()

    @staticmethod
    def to_json_batch(frames: list[TelemetryFrame]) -> str:
        return json.dumps([f.model_dump(mode="json") for f in frames])
