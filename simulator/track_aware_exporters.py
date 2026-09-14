from __future__ import annotations

from .exporters import ParquetTelemetryExporter
from .models import TelemetryFrame


class TrackAwareParquetTelemetryExporter(ParquetTelemetryExporter):
    """Phase-2 telemetry exporter that respects per-track restrictions.

    Legacy restrictions with no track_id continue to apply to every track.
    Manual v11 TSR/maintenance entries always carry an explicit track_id.
    """

    @classmethod
    def _restriction_is_active(cls, restriction, frame: TelemetryFrame) -> bool:
        if restriction.track_id is not None and restriction.track_id != frame.track_id:
            return False
        return super()._restriction_is_active(restriction, frame)
