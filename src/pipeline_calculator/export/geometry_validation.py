"""Validate serialized corridor coordinates with bounded topology checks."""
from __future__ import annotations

import heapq
import math
from pyproj import Geod

GEOD = Geod(ellps='GRS80')
TOPOLOGY_PAIR_BUDGET = 250_000


def _check_topology(xy):
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
    for xmin, xmax, ymin, ymax, i in edges:
        while endings and endings[0][0] < xmin - 1e-8:
            _, expired = heapq.heappop(endings)
            active.pop(expired, None)
        for j, (low, high) in active.items():
            inspections += 1
            if inspections > TOPOLOGY_PAIR_BUDGET:
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


def validated_ring(points):
    cleaned=[]
    for point in points:
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
    for p in cleaned:
        az,_,distance=GEOD.inv(*cleaned[0],*p)
        xy.append((distance*math.sin(math.radians(az)),distance*math.cos(math.radians(az))))
    _check_topology(xy)
    return cleaned,True


def prepare_geometry(section):
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
    for kind,points in candidates:
        try:
            ring,checked=validated_ring(points)
        except (ValueError,TypeError,OverflowError) as exc:
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
        return ring,center,kind,reason
    if bbox_error:
        errors.append(bbox_error)
    raise ValueError('No usable corridor geometry. '+ '; '.join(dict.fromkeys(errors)))
