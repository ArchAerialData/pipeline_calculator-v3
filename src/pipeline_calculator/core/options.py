"""Immutable, non-numeric options captured when an analysis starts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisOptions:
    state_breakdown: bool = False

    def __post_init__(self):
        if not isinstance(self.state_breakdown, bool):
            raise TypeError('state_breakdown must be a boolean')
