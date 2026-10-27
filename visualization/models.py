# visualization/models.py
"""
Typed data models for chart data extracted from LLM responses.

ChartData is the single contract between the extractor (which reads LLM
output) and the renderer (which produces Plotly figures). Neither module
knows about the other's internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ChartType(str, Enum):
    BAR   = "bar"
    LINE  = "line"
    AREA  = "area"
    PIE   = "pie"


@dataclass
class ChartSeries:
    """One named data series within a chart."""
    name:   str
    values: List[float]
    color:  Optional[str] = None   # hex or CSS color; renderer picks default if None

    def __post_init__(self):
        if not self.name:
            raise ValueError("ChartSeries.name must not be empty.")
        if not self.values:
            raise ValueError("ChartSeries.values must not be empty.")
        # Coerce to float defensively (LLM sometimes emits integers)
        self.values = [float(v) for v in self.values]


@dataclass
class ChartData:
    """
    Canonical chart specification produced by the extractor and consumed
    by the renderer.

    Invariants:
    - All series must have the same number of values as len(x).
    - title and x must be non-empty.
    """
    type:            ChartType
    title:           str
    x:               List[str]       # category labels (e.g. fiscal years)
    series:          List[ChartSeries]
    x_label:         Optional[str]  = None
    y_label:         Optional[str]  = None
    currency:        Optional[str]  = None   # e.g. "₹", "$", "€"
    source:          str            = "llm"  # "llm" | "xbrl" — for provenance badge
    xbrl_enriched:   bool           = False  # True when values overridden by XBRL DB

    def __post_init__(self):
        if not self.title:
            raise ValueError("ChartData.title must not be empty.")
        if not self.x:
            raise ValueError("ChartData.x must not be empty.")
        if not self.series:
            raise ValueError("ChartData.series must not be empty.")
        for s in self.series:
            if len(s.values) != len(self.x):
                raise ValueError(
                    f"Series '{s.name}' has {len(s.values)} values "
                    f"but x has {len(self.x)} labels."
                )

    @property
    def value_suffix(self) -> str:
        """Returns currency prefix for axis labels."""
        return self.currency or ""
