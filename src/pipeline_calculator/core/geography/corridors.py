"""Clip all corridor representations to their state before display or export."""
from __future__ import annotations

from shapely.geometry import Polygon
from shapely.ops import unary_union

from pipeline_calculator.core.execution import AnalysisCancelled
from pipeline_calculator.core.corridor_geometry import prepare_corridor
from .boundaries import polygon_parts


def clip_state_corridor(section, state_code, dataset, geod, context=None):
    """Return a copy with authoritative outer/holes polygons, even on failure.

Corridors are display polygons with coordinate-linear edges, matching the
boundary model. They never participate in mileage accounting.
"""
    result = dict(section)
    result["clipped_polygons"] = []
    prepared = prepare_corridor(section, context=context)
    result['diagnostics'] = [dict(d, context={**d.get('context', {}), 'state_code': state_code})
                             for d in prepared['diagnostics']]
    for key in ('visualization_kind', 'visualization_approximation'):
        if key in prepared:
            result[key] = prepared[key]
    # Only the clipped representation may leave this module as prepared geometry.
    result.pop('visualization_polygons', None)
    try:
        if context is not None:
            context.check()
        if prepared['visualization_status'] != 'ready':
            raise ValueError("Corridor has no usable polygon")
        polygon = unary_union([Polygon(p['outer'], p['holes'])
                               for p in prepared['visualization_polygons']])
        boundary = dataset.geometries[state_code]
        clipped = polygon.intersection(boundary)
        if not clipped.is_valid:
            raise ValueError("Clipped corridor polygon is invalid")
        for part in polygon_parts(clipped):
            if context is not None:
                context.check()
            if part.is_empty or part.area <= 0:
                continue
            if not boundary.covers(part):
                # Robust containment check on exact set difference; never buffer
                # the state outward to make an unsafe visualization pass.
                if not part.difference(boundary).is_empty:
                    raise ValueError("Corridor containment verification failed")
            result["clipped_polygons"].append({"outer": [list(p) for p in part.exterior.coords],
                                               "holes": [[list(p) for p in ring.coords] for ring in part.interiors]})
        if not result['clipped_polygons']:
            raise ValueError('Corridor has no polygon area inside this state')
        result['visualization_status'] = 'ready'
    except AnalysisCancelled:
        raise
    except Exception as error:
        result["clipped_polygons"] = []
        result['visualization_status'] = 'omitted'
        result["diagnostics"] = [*[d for d in result['diagnostics']
                                   if d.get('code') != 'corridor_visualization_omitted'],
                                 {"level": "warning", "code": "state_corridor_omitted",
                                  "message": "A state corridor visualization was omitted; mileage is unaffected.",
                                  "context": {"state_code": state_code, "error": str(error)}}]
    return result
