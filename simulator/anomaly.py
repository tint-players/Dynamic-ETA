"""Component B placeholder.

Interactive anomaly injection is intentionally deferred until Component A
(source-to-destination simulation and labelled dataset generation) is stable.
"""


class AnomalyInjector:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Anomaly injection belongs to Component B and is intentionally disabled in Component A v1."
        )
