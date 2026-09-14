"""Clip all corridor representations to their state before display or export."""
from __future__ import annotations

from shapely.geometry import Polygon

from pipeline_calculator.core.execution import AnalysisCancelled
from .boundaries import canonical_geometry, polygon_parts


def clip_state_corridor(section, state_code, dataset, geod, context=None):
    """Return a copy with authoritative outer/holes polygons, even on failure.

Corridors are display polygons with coordinate-linear edges, matching the
boundary model. They never participate in mileage accounting.
"""
    result = dict(section)
    result["clipped_polygons"] = []
    try:
        if context is not None:
            context.check()
        coordinates = section.get("corridor_polygon") or section.get("oriented_polygon")
        if not coordinates or len(coordinates) < 4:
            raise ValueError("Corridor has no usable polygon")
        polygon = Polygon(coordinates)
        if not polygon.is_valid or polygon.is_empty:
            raise ValueError("Corridor polygon is invalid; visualization was omitted")
        polygon = canonical_geometry(polygon)
        boundary = dataset.geometries[state_code]
        clipped = polygon.intersection(boundary)
        if not clipped.is_valid:
            raise ValueError("Clipped corridor polygon is invalid")
        for part in polygon_parts(clipped):
            if context is not None:
                context.check()
            if not boundary.covers(part):
                # Robust containment check on exact set difference; never buffer
                # the state outward to make an unsafe visualization pass.
                if not part.difference(boundary).is_empty:
                    raise ValueError("Corridor containment verification failed")
            result["clipped_polygons"].append({"outer": [list(p) for p in part.exterior.coords],
                                               "holes": [[list(p) for p in ring.coords] for ring in part.interiors]})
    except AnalysisCancelled:
        raise
    except Exception as error:
        result["clipped_polygons"] = []
        result["diagnostics"] = [*section.get("diagnostics", []),
                                 {"level": "warning", "code": "state_corridor_omitted",
                                  "message": "A state corridor visualization was omitted; mileage is unaffected.",
                                  "context": {"state_code": state_code, "error": str(error)}}]
    return result
