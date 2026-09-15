"""Shared corridor candidate selection, bounded validation and canonical polygons.

Geometry decisions do not change measured pipeline lengths or overlap savings.
"""
from __future__ import annotations

import heapq
import math
from pyproj import Geod
from shapely.errors import GEOSException

GEOD = Geod(ellps='GRS80')
TOPOLOGY_PAIR_BUDGET = 250_000
MAX_RING_POINTS = 100_000


def _check_topology(xy, *, budget=TOPOLOGY_PAIR_BUDGET, context=None):
    """Sweep edge bounds; fail to a disclosed fallback if work is excessive.

    The budget counts all active-edge inspections, including disjoint y ranges,
    so adversarial vertical geometry cannot cause an unbounded quadratic scan.
    """
    count = len(xy) - 1
    edges = sorted((min(a[0], b[0]), max(a[0], b[0]),
                    min(a[1], b[1]), max(a[1], b[1]), i)
                   for i, (a, b) in enumerate(zip(xy, xy[1:])))
    active, endings = {}, []
    inspections = 0
    for position, (xmin, xmax, ymin, ymax, i) in enumerate(edges):
        if context is not None and position % 256 == 0:
            context.check()
        while endings and endings[0][0] < xmin - 1e-8:
            _, expired = heapq.heappop(endings)
            active.pop(expired, None)
        for j, (low, high) in active.items():
            inspections += 1
            if inspections > budget:
                raise ValueError('Corridor topology check exceeded its work budget')
            if abs(i-j) == 1 or {i, j} == {0, count-1}:
                continue
            if high < ymin - 1e-8 or low > ymax + 1e-8:
                continue
            if _intersects(xy[i], xy[i+1], xy[j], xy[j+1]):
                raise ValueError('Corridor ring crosses or touches itself')
        active[i] = (ymin, ymax)
        heapq.heappush(endings, (xmax, i))


def coordinate(point):
    lon, lat = (round(float(value), 7) for value in point)
    if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError('Corridor contains non-finite or out-of-range coordinates')
    return lon, lat


def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _intersects(a, b, c, d):
    # Inclusive nonadjacent contacts/collinear overlaps also invalidate a simple ring.
    def on(p, q, r):
        return (abs(_cross(p,q,r)) <= 1e-8 and min(p[0],q[0])-1e-8 <= r[0] <= max(p[0],q[0])+1e-8
                and min(p[1],q[1])-1e-8 <= r[1] <= max(p[1],q[1])+1e-8)
    values = (_cross(a,b,c), _cross(a,b,d), _cross(c,d,a), _cross(c,d,b))
    return values[0]*values[1] < 0 and values[2]*values[3] < 0 or any(
        (on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b)))


def validated_ring(points, *, max_points=MAX_RING_POINTS, topology_budget=TOPOLOGY_PAIR_BUDGET, context=None):
    cleaned=[]
    for position, point in enumerate(points):
        # Bound even duplicate/raw input before cleaning, projection and sorting.
        if context is not None and position % 256 == 0:
            context.check()
        if position >= max_points:
            raise ValueError('Corridor point limit exceeded')
        p=coordinate(point)
        if not cleaned or p!=cleaned[-1]:
            cleaned.append(p)
    if cleaned and cleaned[0]!=cleaned[-1]:
        cleaned.append(cleaned[0])
    if len(set(cleaned))<3:
        raise ValueError('Corridor ring needs three distinct vertices')
    area,_=GEOD.polygon_area_perimeter(*zip(*cleaned))
    if not math.isfinite(area) or abs(area)<1e-4:
        raise ValueError('Corridor ring has no usable area')
    xy=[]
    for position, p in enumerate(cleaned):
        if context is not None and position % 256 == 0:
            context.check()
        az,_,distance=GEOD.inv(*cleaned[0],*p)
        xy.append((distance*math.sin(math.radians(az)),distance*math.cos(math.radians(az))))
    _check_topology(xy, budget=topology_budget, context=context)
    return cleaned,True


def prepare_geometry(section, *, validator=None, context=None):
    """Compatibility tuple for ordinary previews, backed by the common selector."""
    return _select_geometry(section, validator=validator, context=context)[:4]


def _select_geometry(section, *, validator=None, context=None):
    if validator is None:
        validator = lambda points: validated_ring(points, context=context)
    candidates=[]
    if section.get('corridor_polygon'):
        candidates.append((section.get('corridor_geometry_kind','sampled_curve'),section['corridor_polygon']))
    if section.get('oriented_polygon'):
        candidates.append(('oriented_rectangle',section['oriented_polygon']))
    bbox=section.get('bbox')
    bbox_error = None
    if bbox:
        try:
            a,b,c,d=(bbox[k] for k in ('min_lon','max_lon','min_lat','max_lat'))
            candidates.append(('bounding_rectangle',[(a,c),(b,c),(b,d),(a,d),(a,c)]))
        except (KeyError,TypeError):
            bbox_error = 'Corridor bounding rectangle is incomplete'
    errors=[]
    attempted = set()
    for kind,points in candidates:
        if id(points) in attempted:
            continue
        attempted.add(id(points))
        try:
            ring,checked=validator(points)
            polygons = normalize_polygons([{'outer': ring, 'holes': []}], context=context)
        except (ValueError,TypeError,OverflowError,GEOSException) as exc:
            errors.append(str(exc))
            continue
        if 'center_lon' in section or 'center_lat' in section:
            try:
                center=coordinate((section['center_lon'],section['center_lat']))
            except (KeyError,ValueError,TypeError,OverflowError) as exc:
                raise ValueError('Corridor center is invalid') from exc
        else:
            center=ring[0]
        reason='Approximate visualization from sampled pipeline segments; not a surveyed boundary.'
        if kind != 'sampled_curve':
            reason+=' Rectangle approximation may include areas outside the qualified paths.'
        if errors:
            reason+=' Preferred geometry was invalid: '+ '; '.join(dict.fromkeys(errors))+'.'
        if not checked:
            reason+=' Detailed topology has not been validated.'
        return ring,center,kind,reason,polygons,errors
    if bbox_error:
        errors.append(bbox_error)
    raise ValueError('No usable corridor geometry. '+ '; '.join(dict.fromkeys(errors)))


def normalize_polygons(polygons, *, context=None):
    """Validate and split coordinate-linear rings before planar predicates.

    Do not round already clipped coordinates: rounding may move a state edge.
    No buffer, snapping or topology repair is permitted here.
    """
    from shapely.affinity import translate
    from shapely.geometry import Polygon, box

    count = 0

    def ring(points):
        nonlocal count
        cleaned = []
        for point in points:
            count += 1
            if count > MAX_RING_POINTS:
                raise ValueError('Corridor point limit exceeded')
            if context is not None and count % 256 == 1:
                context.check()
            lon, lat = map(float, point[:2])
            if not (math.isfinite(lon) and math.isfinite(lat)
                    and -180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError('Corridor contains non-finite or out-of-range coordinates')
            if not cleaned or (lon, lat) != cleaned[-1]:
                cleaned.append((lon, lat))
        if len(set(cleaned)) < 3:
            raise ValueError('Corridor ring needs three distinct vertices')
        if cleaned[0] != cleaned[-1]:
            cleaned.append(cleaned[0])
        return cleaned

    def unwrap(points, center=None):
        shifted = [points[0]]
        for lon, lat in points[1:]:
            shifted.append((lon + 360*round((shifted[-1][0]-lon)/360), lat))
        if center is not None:
            offset = 360*round((center-sum(p[0] for p in shifted)/len(shifted))/360)
            if offset:
                shifted = [(lon+offset, lat) for lon, lat in shifted]
        return shifted

    def parts(shape):
        if shape.geom_type == 'Polygon':
            yield shape
        elif hasattr(shape, 'geoms'):
            for item in shape.geoms:
                yield from parts(item)

    result = []
    for polygon_index, spec in enumerate(polygons):
        if polygon_index >= MAX_RING_POINTS:
            raise ValueError('Corridor polygon limit exceeded')
        outer = ring(spec['outer'])
        holes = [ring(hole) for hole in spec.get('holes', [])]
        crosses_dateline = any(abs(a[0]-b[0]) > 180
                               for points in [outer, *holes] for a, b in zip(points, points[1:]))
        if crosses_dateline:
            # Integer offsets preserve input coordinates in their existing zone;
            # a radians/degrees roundtrip can invalidate certified containment.
            outer = unwrap(outer)
            if max(p[0] for p in outer)-min(p[0] for p in outer) > 360:
                raise ValueError('Corridor winds around the globe; visualization is ambiguous')
            center = sum(p[0] for p in outer)/len(outer)
            holes = [unwrap(hole, center) for hole in holes]
        polygon = Polygon(outer, holes)
        if polygon.is_empty or not polygon.is_valid:
            raise ValueError('Corridor polygon is invalid')
        if crosses_dateline:
            lo, _, hi, _ = polygon.bounds
            normalized_parts = []
            for zone in range(math.floor((lo+180)/360), math.floor((hi+180)/360)+1):
                if context is not None:
                    context.check()
                piece = polygon.intersection(box(-180+360*zone, -90, 180+360*zone, 90))
                normalized_parts.extend(parts(translate(piece, xoff=-360*zone) if zone else piece))
        else:
            # Already canonical clipped coordinates remain bit-for-bit unchanged.
            normalized_parts = [polygon]
        for part in normalized_parts:
            if part.is_empty or part.area <= 0:
                continue
            if not part.is_valid:
                raise ValueError('Corridor polygon is invalid after longitude normalization')
            result.append({'outer': [list(p) for p in part.exterior.coords],
                           'holes': [[list(p) for p in hole.coords] for hole in part.interiors]})
    if not result:
        raise ValueError('No usable corridor polygon geometry')
    return result


def _visual_diagnostic(code, message, **context):
    return {'level': 'warning' if code.endswith('omitted') else 'info',
            'code': code, 'message': message, 'context': context}


def prepare_corridor(section, *, require_clipped=False, context=None):
    """Return a JSON-native visualization decision without mutating the section.

    Prepared or clipped polygons are authoritative. In particular an empty
    clipped result must never expose an original uncut fallback during export.
    """
    from pipeline_calculator.core.execution import AnalysisCancelled

    result = dict(section)
    diagnostics = list(section.get('diagnostics') or [])
    result['diagnostics'] = diagnostics
    try:
        if context is not None:
            context.check()
        key = ('clipped_polygons' if 'clipped_polygons' in section else
               'visualization_polygons' if 'visualization_polygons' in section else None)
        if require_clipped and key != 'clipped_polygons':
            raise ValueError('State corridor lacks boundary-clipped geometry')
        if key is not None:
            polygons = normalize_polygons(section[key], context=context)
            kind = section.get('visualization_kind', 'state_clipped' if key == 'clipped_polygons' else 'prepared')
            reason = section.get('visualization_approximation',
                                 'Approximate visualization; not a surveyed boundary.')
        else:
            _, _, kind, reason, polygons, errors = _select_geometry(section, context=context)
            if kind != 'sampled_curve' or errors:
                diagnostics.append(_visual_diagnostic(
                    'corridor_geometry_fallback',
                    'A corridor uses a rectangle approximation; mileage is unaffected.',
                    selected_geometry=kind, rejected_candidates=list(dict.fromkeys(errors))))
        result.update(visualization_polygons=polygons, visualization_status='ready',
                      visualization_kind=kind, visualization_approximation=reason)
    except AnalysisCancelled:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, GEOSException) as error:
        result.update(visualization_polygons=[], visualization_status='omitted')
        if not any(str(d.get('code', '')).endswith('omitted') or
                   d.get('code') == 'state_corridor_unavailable' for d in diagnostics):
            diagnostics.append(_visual_diagnostic(
                'corridor_visualization_omitted',
                'A corridor visualization was omitted; mileage is unaffected.', error=str(error)))
    return result


def prepare_scope_visualizations(scope, *, state_code=None, context=None):
    """Copy only changed containers; expose decisions in scope diagnostics too."""
    result = dict(scope)
    diagnostics = list(scope.get('diagnostics') or [])
    result['diagnostics'] = diagnostics
    overlap = scope.get('overlap_analysis')
    if not isinstance(overlap, dict):
        return result
    result['overlap_analysis'] = prepared_overlap = dict(overlap)
    sections = []
    for index, section in enumerate(overlap.get('bundled_sections', [])):
        prepared = prepare_corridor(section, require_clipped=state_code is not None, context=context)
        scoped = []
        for diagnostic in prepared['diagnostics']:
            detail = dict(diagnostic)
            detail['context'] = dict(diagnostic.get('context') or {})
            detail['context'].update(section_index=index, scope=state_code or 'Combined')
            scoped.append(detail)
            if diagnostic != detail and diagnostic in diagnostics:
                diagnostics.remove(diagnostic)
            if detail not in diagnostics:
                diagnostics.append(detail)
        prepared['diagnostics'] = scoped
        sections.append(prepared)
    prepared_overlap['bundled_sections'] = sections
    return result
