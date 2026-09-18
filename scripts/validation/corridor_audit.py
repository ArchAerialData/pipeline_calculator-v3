"""Independent complete KML polygon parsing and fixed-radius display checks.

No production geometry builders are called. Reference paths retain native bends,
are sampled on GRS80, and use a separate dense local-plane buffer with 128 arc
segments per quadrant. This audit explicitly bounds its local chart domain.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from pyproj import Geod
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

GEOD = Geod(ellps='GRS80')
NS = {'k': 'http://www.opengis.net/kml/2.2'}


def document_polygons(document):
    """Read every exterior/interior ring; reject malformed or empty maps."""
    root = ET.fromstring(document) if isinstance(document, (str, bytes)) else document
    polygons = []
    for node in root.findall('.//k:Polygon', NS):
        boundaries = node.findall('k:outerBoundaryIs/k:LinearRing/k:coordinates', NS)
        if len(boundaries) != 1:
            raise ValueError('A polygon must have exactly one outer ring')
        boundaries += node.findall('k:innerBoundaryIs/k:LinearRing/k:coordinates', NS)
        rings = []
        for boundary in boundaries:
            coordinates = [tuple(map(float, token.split(',')[:2])) for token in (boundary.text or '').split()]
            if (len(coordinates) < 4 or coordinates[0] != coordinates[-1] or
                    any(len(point) != 2 or not all(map(math.isfinite, point)) or
                        not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90 for point in coordinates)):
                raise ValueError('Malformed, open, or out-of-range polygon ring')
            rings.append(coordinates)
        polygons.append({'outer': rings[0], 'holes': rings[1:]})
    if not polygons:
        raise ValueError('No polygon geometry in map')
    return polygons


def dense_path(coordinates, maximum_step=2.0):
    result = [tuple(coordinates[0])]
    for a, b in zip(coordinates, coordinates[1:]):
        bearing, _, length = GEOD.inv(*a, *b)
        if length == 0:
            continue
        count = max(1, math.ceil(length / maximum_step))
        result.extend(GEOD.fwd(*a, bearing, length * i / count)[:2] for i in range(1, count))
        result.append(tuple(b))
    return result


def project(coordinates, origin):
    points = []
    for coordinate in coordinates:
        bearing, _, length = GEOD.inv(*origin, *coordinate)
        bearing = math.radians(bearing)
        points.append((length * math.sin(bearing), length * math.cos(bearing)))
    return points


def dense_ring(coordinates, maximum_step=2.0):
    """KML polygon edges follow the serialized coordinate-linear boundary model."""
    result = [coordinates[0]]
    for a, b in zip(coordinates, coordinates[1:]):
        longitude = b[0] + 360 * round((a[0] - b[0]) / 360)
        length = abs(GEOD.inv(*a, *b)[2])
        count = max(1, math.ceil(length / maximum_step))
        result.extend((a[0] + (longitude-a[0])*i/count, a[1]+(b[1]-a[1])*i/count)
                      for i in range(1, count))
        result.append(b)
    return result


def metric_polygons(polygons, origin):
    shapes = []
    for polygon in polygons:
        shape = Polygon(project(dense_ring(polygon['outer']), origin),
                        [project(dense_ring(hole), origin) for hole in polygon['holes']])
        if not shape.is_valid or shape.is_empty or shape.area <= 0:
            raise ValueError('Invalid serialized polygon, component, or hole')
        shapes.append(shape)
    return shapes


def consecutive_ranges(indices):
    result = []
    for value in sorted(set(indices)):
        if result and value == result[-1][1]:
            result[-1][1] += 1
        else:
            result.append([value, value+1])
    return result


def path_span(coordinates, lower, upper):
    """Independent cumulative-chainage slice retaining every intervening vertex."""
    lengths = [abs(GEOD.inv(*a, *b)[2]) for a, b in zip(coordinates, coordinates[1:])]
    total = math.fsum(lengths)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper <= total+1e-7):
        raise ValueError('Qualified span exceeds source path')
    upper = min(upper, total)
    result, offset = [], 0.0
    for a, b, length in zip(coordinates, coordinates[1:], lengths):
        if length > 0 and offset <= upper and offset+length >= lower:
            bearing = GEOD.inv(*a, *b)[0]
            start, end = max(lower-offset, 0.0), min(upper-offset, length)
            if start < end:
                first = a if start == 0 else GEOD.fwd(*a, bearing, start)[:2]
                last = b if end == length else GEOD.fwd(*a, bearing, end)[:2]
                if not result or result[-1] != first:
                    result.append(first)
                result.append(last)
        offset += length
    if len(result) < 2:
        raise ValueError('No positive source span')
    return result


def section_runs(pipelines, qualified, step):
    paths, metadata = [], []
    for pipe, path, ids in zip(qualified['pair'], qualified['paths'], qualified['segment_ids']):
        pipeline = pipelines[pipe]
        source = pipeline['coordinate_paths'][path]
        indices = [pipeline['segments'][index]['path_segment_index'] for index in ids]
        for first, last in consecutive_ranges(indices):
            paths.append(path_span(source, first*step, last*step))
            metadata.append({'source_id': pipeline.get('id', pipe), 'scope_path_index': path,
                             'sample_start': first, 'sample_end': last, 'start_m': first*step, 'end_m': last*step})
    return paths, metadata


def inspect_document(document, generating_paths, *, padding_m=None, target_m=.05, max_radius_m=25_000, clip_geometry=None):
    polygons = document if isinstance(document, list) else document_polygons(document)
    if not generating_paths:
        raise ValueError('No independently identified generating paths')
    origin = generating_paths[0][0]
    paths = [project(dense_path(path), origin) for path in generating_paths]
    maximum_radius = max(math.hypot(x, y) for path in paths for x, y in path)
    if maximum_radius > max_radius_m:
        raise ValueError(f'Independent local audit radius exceeded: {maximum_radius:g} m')
    parts = metric_polygons(polygons, origin)
    actual = unary_union(parts)
    lines = unary_union([LineString(path) for path in paths])
    uncovered = lines.difference(actual.buffer(target_m)).length
    result = {'part_count': len(polygons), 'hole_count': sum(len(p['holes']) for p in polygons),
              'vertex_count': sum(len(p['outer'])+sum(map(len, p['holes'])) for p in polygons),
              'valid_polygon': True, 'area_m2': actual.area, 'origin': origin,
              'maximum_reference_radius_m': maximum_radius,
              'uncovered_source_length_m': uncovered,
              'generating_run_count': len(paths), 'source_coverage_passed': uncovered <= 1e-7,
              'reference_step_m': 2.0, 'target_m': target_m, 'padding_m': padding_m}
    if padding_m is not None:
        reference = lines.buffer(padding_m, cap_style='round', join_style='round', quad_segs=128)
        if clip_geometry is not None:
            # Crop the native state before dense metric conversion. This is a
            # validation workload bound, not a buffered/shifted state boundary.
            points = [point for p in polygons for point in p['outer']]
            points += [point for path in generating_paths for point in path]
            west, east = min(p[0] for p in points), max(p[0] for p in points)
            south, north = min(p[1] for p in points), max(p[1] for p in points)
            local = clip_geometry.intersection(box(west-.001, south-.001, east+.001, north+.001))
            def parts_of(shape):
                if shape.geom_type == 'Polygon':
                    yield shape
                elif hasattr(shape, 'geoms'):
                    for part in shape.geoms:
                        yield from parts_of(part)
            specs = [{'outer': list(p.exterior.coords), 'holes': [list(r.coords) for r in p.interiors]}
                     for p in parts_of(local)]
            reference = reference.intersection(unary_union(metric_polygons(specs, origin)))
        missing = reference.buffer(-target_m).difference(actual)
        excess = actual.difference(reference.buffer(target_m))
        result.update(reference_area_m2=reference.area,
                      missing_inner_neighborhood_m2=missing.area, excess_outer_neighborhood_m2=excess.area,
                      inner_radius_passed=missing.is_empty, outer_radius_passed=excess.is_empty,
                      boundary_hausdorff_m=actual.boundary.hausdorff_distance(reference.boundary))
    result['passed'] = result['source_coverage_passed'] and result.get('inner_radius_passed', True) and result.get('outer_radius_passed', True)
    return result, parts, paths
