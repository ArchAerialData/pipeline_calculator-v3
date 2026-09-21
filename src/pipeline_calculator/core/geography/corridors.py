"""Clip all corridor representations to their state before display or export."""
from __future__ import annotations

from shapely import get_num_coordinates
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

from pipeline_calculator.core.execution import AnalysisCancelled
from pipeline_calculator.core.corridor_geometry import prepare_corridor
from .boundaries import polygon_parts


MAX_CLIP_VERTICES = 4096
MAX_CLIP_PARTS = 4096


class _StateCorridorLimit(ValueError):
    """A structured resource reason, distinct from invalid state geometry."""

    code = 'corridor_buffer_limit'


def _vertices(shape):
    return int(get_num_coordinates(shape))


def _limits(budget):
    return (min(MAX_CLIP_VERTICES, budget.max_native_vertices),
            min(32, budget.max_native_operands)) if budget is not None else (MAX_CLIP_VERTICES, 32)


def _boundary_covers(boundary, shape, context, budget):
    """Immutable boundary queries have a separate, explicitly bounded budget."""
    if context is not None:
        context.check()
    if budget is not None:
        budget.boundary_query(_vertices(boundary))
    result = boundary.covers(shape)
    if context is not None:
        context.check()
    return result


def _outside_is_empty(shape, boundary, context, budget):
    if context is not None:
        context.check()
    if budget is not None:
        budget.boundary_query(_vertices(boundary))
    # This final check deliberately uses the complete immutable reference. Local
    # window clipping cannot weaken the strict published containment contract.
    difference = shape.difference(boundary)
    if context is not None:
        context.check()
    if _vertices(difference) > _limits(budget)[0]:
        raise _StateCorridorLimit('Boundary containment check output exceeds its work limit')
    return difference.is_empty


def _local_boundary(shape, state_code, dataset, context, budget=None):
    """Retain original boundary edge endpoints in a bounded local mask.

    Shortening a boundary edge at a query box can change its slope by an ULP and
    make the final clipped polygon leak. Join open ends outside the query instead;
    all boundary segments which meet the corridor retain their original support.
    """
    from collections import Counter
    from shapely.geometry import MultiLineString, Point
    from shapely.strtree import STRtree

    boundary = dataset.geometries[state_code]
    cap, operands = _limits(budget)
    if operands < 1:
        raise _StateCorridorLimit('State corridor native operand limit exceeded')
    if _vertices(boundary) <= cap:
        return boundary
    lo, bottom, hi, top = shape.bounds
    margin = max(1e-8, (hi-lo)*1e-6, (top-bottom)*1e-6)
    window = box(max(-180, lo-margin), max(-90, bottom-margin),
                 min(180, hi+margin), min(90, top+margin))
    lines, ends = [], Counter()
    for code, edge in dataset.candidate_edges(window.bounds, context=context):
        if code != state_code:
            continue
        if context is not None:
            context.check()
        if 2*(len(lines)+1) + 5 > cap:
            raise _StateCorridorLimit('Local state boundary exceeds the corridor clipping work limit')
        segment = LineString(edge)
        if not segment.intersects(window):
            continue
        points = [tuple(p) for p in edge]
        lines.append(points)
        ends.update(points)
    if not lines:
        return window if _boundary_covers(boundary, window.representative_point(), context, budget) else Polygon()
    all_points = [point for line in lines for point in line]
    west = min(lo-margin, min(p[0] for p in all_points))-margin
    east = max(hi+margin, max(p[0] for p in all_points))+margin
    south = min(bottom-margin, min(p[1] for p in all_points))-margin
    north = max(top+margin, max(p[1] for p in all_points))+margin
    frame = box(west, south, east, north)
    center = ((lo+hi)/2, (bottom+top)/2)
    original_edges = [LineString(points) for points in lines]
    edge_tree = STRtree(original_edges)
    for point, degree in ends.items():
        if degree % 2 == 0:
            continue
        # Every open endpoint is outside the query; its outward continuation
        # cannot intersect the corridor inside it.
        if window.contains(Point(point)):
            raise ValueError('Local state boundary has an unconnected interior vertex')
        dx, dy = point[0]-center[0], point[1]-center[1]
        tx = ((east if dx > 0 else west)-center[0])/dx if dx else float('inf')
        ty = ((north if dy > 0 else south)-center[1])/dy if dy else float('inf')
        if tx <= ty:
            end = (east if dx > 0 else west, center[1]+dy*tx)
        else:
            end = (center[0]+dx*ty, north if dy > 0 else south)
        connector = LineString((point, end))
        for candidate in edge_tree.query(connector):
            edge = original_edges[int(candidate)]
            contact = connector.intersection(edge)
            if not contact.is_empty and (contact.geom_type != 'Point'
                    or tuple(contact.coords[0]) not in (tuple(edge.coords[0]), tuple(edge.coords[-1]))):
                raise ValueError('Local state framing would alter an original boundary edge')
        lines.append([point, end])
    lines.append(list(frame.exterior.coords))
    count = sum(map(len, lines))
    segments = [LineString((a, b)) for line in lines for a, b in zip(line, line[1:])]
    tree = STRtree(segments)
    candidates = 0
    for index, segment in enumerate(segments):
        if context is not None:
            context.check()
        hits = tree.query(segment)
        if budget is not None:
            budget.charge(len(hits))
        candidates += sum(int(other) > index for other in hits)
        if count + 4*candidates > cap:
            raise _StateCorridorLimit('State boundary noding exceeds the corridor clipping work limit')
    expansion = count + 4*candidates
    if expansion > cap:
        raise _StateCorridorLimit('State boundary noding exceeds the corridor clipping work limit')
    if budget is not None:
        budget.preflight(count+expansion)
        budget.charge(count)
    if context is not None:
        context.check()
    # One bounded linework operand, with explicit output capacity.
    noded = unary_union(MultiLineString(lines))
    if _vertices(noded) > cap:
        raise _StateCorridorLimit('State boundary noding output exceeds its work limit')
    if budget is not None:
        budget.charge(_vertices(noded))
    faces = []
    for face in polygonize(noded):
        if context is not None:
            context.check()
        if _vertices(face) > cap:
            raise _StateCorridorLimit('Local state face exceeds its work limit')
        sample = _bounded_intersection(face, window, context, budget)
        if (not sample.is_empty and sample.area > 0
                and _boundary_covers(boundary, sample.representative_point(), context, budget)):
            faces.append(face)
        if len(faces) > operands:
            raise _StateCorridorLimit('Local state boundary has too many faces')
    count = sum(_vertices(face) for face in faces)
    if count > cap:
        raise _StateCorridorLimit('Local state boundary exceeds the corridor clipping work limit')
    if budget is not None:
        budget.preflight(count*2)
        budget.charge(count)
    result = unary_union(faces)
    if _vertices(result) > cap:
        raise _StateCorridorLimit('Local state mask output exceeds its work limit')
    if budget is not None:
        budget.charge(_vertices(result))
    return result


def _bounded_intersection(polygon, mask, context, budget=None):
    """Preflight overlay input and a conservative crossing/output bound."""
    from shapely.strtree import STRtree
    cap, operands = _limits(budget)
    if operands < 2:
        raise _StateCorridorLimit('State corridor native operand limit exceeded')
    edges = []
    for part in polygon_parts(mask):
        for ring in (part.exterior, *part.interiors):
            edges.extend(LineString((a, b)) for a, b in zip(ring.coords, list(ring.coords)[1:]))
    count = _vertices(polygon) + _vertices(mask)
    if count > cap:
        raise _StateCorridorLimit('State corridor overlay input exceeds its work limit')
    tree = STRtree(edges)
    crossing_bound = 0
    for ring in (polygon.exterior, *polygon.interiors):
        for index, (a, b) in enumerate(zip(ring.coords, list(ring.coords)[1:])):
            if context is not None and index % 256 == 0:
                context.check()
            crossing_bound += len(tree.query(LineString((a, b))))
            if count + 4*crossing_bound > cap:
                raise _StateCorridorLimit('State corridor overlay expansion exceeds its work limit')
    if budget is not None:
        budget.preflight(2*count + 4*crossing_bound)
        budget.charge(count)
    if context is not None:
        context.check()
    result = polygon.intersection(mask)
    if context is not None:
        context.check()
    if _vertices(result) > cap:
        raise _StateCorridorLimit('State corridor overlay output exceeds its work limit')
    if budget is not None:
        budget.charge(_vertices(result))
    return result


def clip_state_corridor(section, state_code, dataset, geod, context=None, *, budget=None,
                        replace_retained=False):
    """Return a copy with authoritative outer/holes polygons, even on failure.

Corridors are display polygons with coordinate-linear edges, matching the
boundary model. They never participate in mileage accounting.
"""
    if budget is None:
        from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
        budget = CorridorGeometryBudget()
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
        if replace_retained and section.get('visualization_status') == 'ready':
            prior = section.get('clipped_polygons', section.get('visualization_polygons', []))
            budget.release_retained(sum(len(p['outer']) + sum(len(h) for h in p['holes']) for p in prior))
        if prepared['visualization_status'] != 'ready':
            raise ValueError("Corridor has no usable polygon")
        boundary = dataset.geometries[state_code]
        cap, operands = _limits(budget)
        pieces = []
        part_limit = min(MAX_CLIP_PARTS, budget.max_parts)
        ring_limit = min(8192, budget.max_rings)
        output_limit = min(100_000, budget.max_section_points)
        refinement_passes = 0
        output_vertices = output_rings = 0
        for spec in prepared['visualization_polygons']:
            if context is not None:
                context.check()
            if len(spec['outer']) + sum(len(h) for h in spec['holes']) > cap:
                raise _StateCorridorLimit('State corridor polygon exceeds its clipping work limit')
            polygon = Polygon(spec['outer'], spec['holes'])
            if _vertices(polygon) > cap or operands < 1:
                raise _StateCorridorLimit('State corridor polygon exceeds its clipping work limit')
            if budget is not None:
                budget.charge(_vertices(polygon))
            if _boundary_covers(boundary, polygon, context, budget):
                clipped = polygon
                mask = None
            else:
                mask = _local_boundary(polygon, state_code, dataset, context, budget)
                clipped = _bounded_intersection(polygon, mask, context, budget)
            if not clipped.is_valid:
                raise ValueError('Clipped corridor polygon is invalid')
            pending = [(part, 0) for part in polygon_parts(clipped)]
            while pending:
                part, attempt = pending.pop()
                if context is not None:
                    context.check()
                if (_boundary_covers(boundary, part, context, budget)
                        or _outside_is_empty(part, boundary, context, budget)):
                    pieces.append(part)
                    output_vertices += _vertices(part)
                    output_rings += 1 + len(part.interiors)
                    if (len(pieces) > part_limit
                            or output_rings > ring_limit or output_vertices > output_limit):
                        raise _StateCorridorLimit('State corridor output exceeds its part/ring/vertex limits')
                    continue
                if mask is None or attempt >= 8:
                    raise ValueError('Corridor containment verification failed')
                # GEOS can leave an ULP-sized outside wedge even when clipping
                # directly against the full original state. Refine with the same
                # exact set intersection, never snap a point or expand the state.
                # No nonempty outside difference is accepted after these retries.
                refined = _bounded_intersection(part, mask, context, budget)
                if not refined.is_valid or refined.is_empty:
                    raise ValueError('Corridor containment refinement failed')
                # Refinement may only remove floating-point overlay residue.
                # One nanodegree is <0.12 mm everywhere, well below the map's
                # error budget. Check full area as well as boundary displacement.
                roundoff_degrees = 1e-9
                if _vertices(part) + _vertices(refined) > cap:
                    raise _StateCorridorLimit('Corridor containment refinement changed more than numerical roundoff')
                if (part.hausdorff_distance(refined) > roundoff_degrees
                        or abs(part.area-refined.area) > part.length*roundoff_degrees):
                    raise ValueError('Corridor containment refinement changed more than numerical roundoff')
                refinement_passes += 1
                pending.extend((piece, attempt+1) for piece in polygon_parts(refined))
                if len(pending) + len(pieces) > part_limit:
                    raise _StateCorridorLimit('State corridor has too many polygon parts')
            if len(pieces) > part_limit:
                raise _StateCorridorLimit('State corridor has too many polygon parts')
        for part in pieces:
            if context is not None:
                context.check()
            if part.is_empty or part.area <= 0:
                continue
            if not _boundary_covers(boundary, part, context, budget):
                # Robust containment check on exact set difference; never buffer
                # the state outward to make an unsafe visualization pass.
                if not _outside_is_empty(part, boundary, context, budget):
                    raise ValueError("Corridor containment verification failed")
            result["clipped_polygons"].append({"outer": [list(p) for p in part.exterior.coords],
                                               "holes": [[list(p) for p in ring.coords] for ring in part.interiors]})
        if not result['clipped_polygons']:
            raise ValueError('Corridor has no polygon area inside this state')
        if budget is not None:
            budget.retain(sum(_vertices(p) for p in pieces))
        result['visualization_status'] = 'ready'
        if 'visualization_metadata' in result:
            result['visualization_metadata'] = dict(result['visualization_metadata'],
                state_code=state_code, clipping_status='clipped',
                clipping_refinement_passes=refinement_passes,
                part_count=len(result['clipped_polygons']),
                hole_count=sum(len(p['holes']) for p in result['clipped_polygons']),
                vertex_count=sum(len(p['outer']) + sum(len(h) for h in p['holes'])
                                 for p in result['clipped_polygons']))
            # Compatibility alias is only the full, clipped, hole-free shape.
            result['corridor_polygon'] = (result['clipped_polygons'][0]['outer']
                if len(result['clipped_polygons']) == 1 and not result['clipped_polygons'][0]['holes'] else [])
    except AnalysisCancelled:
        raise
    except Exception as error:
        result["clipped_polygons"] = []
        if 'visualization_schema_version' in result:
            result['corridor_polygon'] = []
            if isinstance(result.get('visualization_metadata'), dict):
                result['visualization_metadata'] = dict(result['visualization_metadata'],
                    part_count=0, hole_count=0, vertex_count=0, clipping_status='omitted')
        result['visualization_status'] = 'omitted'
        result["diagnostics"] = [*[d for d in result['diagnostics']
                                   if d.get('code') != 'corridor_visualization_omitted'],
                                 {"level": "warning", "code": "state_corridor_omitted",
                                  "message": "A state corridor visualization was omitted; mileage is unaffected.",
                                  "context": {"state_code": state_code, "error": str(error),
                                              "reason_code": getattr(error, 'code', 'state_corridor_geometry_invalid')}}]
    return result
