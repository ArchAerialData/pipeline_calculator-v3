"""Immutable, non-numeric options captured when an analysis starts."""
from dataclasses import dataclass, field
from pipeline_calculator.core.corridor_buffer import CorridorDisplayOptions


@dataclass(frozen=True)
class AnalysisOptions:
    state_breakdown: bool = False
    corridor_display: CorridorDisplayOptions = field(default_factory=CorridorDisplayOptions)

    def __post_init__(self):
        if not isinstance(self.state_breakdown, bool):
            raise TypeError('state_breakdown must be a boolean')
        if not isinstance(self.corridor_display, CorridorDisplayOptions):
            raise TypeError('corridor_display must be immutable CorridorDisplayOptions')
