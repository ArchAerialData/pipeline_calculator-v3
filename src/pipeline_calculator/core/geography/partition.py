"""Geodesic path interval partitioning against coordinate-linear state edges.

Each bounded source subedge is radial (therefore exactly straight) in an
ellipsoidal azimuthal-equidistant projection centered at its start. Boundary
curves are adaptively sampled there and crossing roots refined on the original
coordinate-linear boundary. Distances always come from the original geodesic,
never a projected line length. No positive-length interval is discarded.
"""
from __future__ import annotations

import math
from collections import defaultdict

from scipy.optimize import brentq, minimize_scalar
from shapely.geometry import LineString, box

from pipeline_calculator.core.coordinates import coordinate_paths_for_pipeline
from pipeline_calculator.core.execution import AnalysisCancelled
from .boundaries import load_boundaries

MAX_CHUNK_METERS = 20_000.0
MAX_WORK_ITEMS = 2_000_000
PROJECTION_ERROR_METERS = 0.001
COINCIDENCE_ROUNDOFF_METERS = 1e-8
ROOT_DISTANCE_METERS = 1e-6


class _NumericalClippingError(ValueError):
    """An operation could not establish ownership within its numerical bound."""


def _checkpoint(context):
    if context is not None:
        context.check()


class _GeodesicProjection:
    """Ellipsoidal AEQD from inverse geodesic distance and azimuth.

    The definition x=d*sin(azimuth), y=d*cos(azimuth) uses the actual source
    origin and the analyzer's ellipsoid. It avoids CRS parameter serialization
    and the general AEQD transform's near-origin zero-distance shortcut, which
    can erase real submillimeter crossings.
    """

    def __init__(self, lon, lat, geod):
        self.origin = (lon, lat)
        self.geod = geod

    def transform(self, lon, lat):
        azimuth, _, distance = self.geod.inv(*self.origin, lon, lat)
        angle = math.radians(azimuth)
        return distance*math.sin(angle), distance*math.cos(angle)


def _local_projection(lon, lat, geod):
    return _GeodesicProjection(lon, lat, geod)


def _longitude_near(lon, center):
    return lon + 360 * round((center-lon) / 360)


def _shared_stations(start, finish, a, b, length, geod):
    """Prove coincidence and retain exact endpoint identities, without snapping.

    Boundary edges are coordinate-linear and source edges are geodesics. Their
    positive-length common arcs are meridians or the equator. A merely close
    projected chord is insufficient proof, especially for very short edges.
    Geographic endpoints define the overlap; projection roundoff cannot create
    another event immediately before an already known source endpoint.
    """
    finish = (_longitude_near(finish[0], start[0]), finish[1])
    if start[0] == finish[0] == a[0] == b[0]:
        axis = 1
    elif start[1] == finish[1] == a[1] == b[1] == 0:
        axis = 0
    else:
        return None
    lower = max(min(start[axis], finish[axis]), min(a[axis], b[axis]))
    upper = min(max(start[axis], finish[axis]), max(a[axis], b[axis]))
    if upper <= lower:
        return None

    def station(value):
        if value == start[axis]:
            return 0.0
        if value == finish[axis]:
            return length
        point = (start[0], value) if axis == 1 else (value, 0.0)
        return abs(geod.inv(*start, *point)[2])

    lo, hi = sorted((station(lower), station(upper)))
    if not 0 <= lo < hi <= length:
        raise _NumericalClippingError("Distinct shared endpoints could not be measured reliably")
    return lo, hi


def _edge_events(start, finish, azimuth, length, dataset, geod, context, budget):
    transformer = None
    theta = math.radians(azimuth)
    along = (math.sin(theta), math.cos(theta))
    normal = (-along[1], along[0])

    # Geodesic source samples bound the local geographic query. Locate a
    # latitude extremum only when endpoint bearings prove one exists; normal
    # short vertex-dense inputs need neither optimization nor a projection.
    def source_point(distance):
        lon, lat, _ = geod.fwd(*start, azimuth, distance)
        return _longitude_near(lon, start[0]), lat

    points = [source_point(length * fraction / 8) for fraction in range(9)]
    back_azimuth = geod.inv(*start, *finish)[1]
    if math.cos(theta)*math.cos(math.radians(back_azimuth)) > 0:
        sign = -1 if math.cos(theta) > 0 else 1
        extremum = minimize_scalar(lambda d: sign*source_point(d)[1], bounds=(0, length), method="bounded")
        points.append(source_point(float(extremum.x)))
    pad = 1e-8  # Query expansion only. Never used for state membership or snapping.
    west = min(p[0] for p in points)-pad
    east = max(p[0] for p in points)+pad
    south = max(-90, min(p[1] for p in points)-pad)
    north = min(90, max(p[1] for p in points)+pad)
    if east-west > 180 or north >= 89.999999 or south <= -89.999999:
        raise ValueError("Polar or longitude-ambiguous path interval is outside the supported numerical projection domain")

    events = [0.0, length]
    shared = []
    seen = set()
    for zone in range(math.floor((west+180)/360), math.floor((east+180)/360)+1):
        query_bounds = (max(-180.0, west-360*zone), south, min(180.0, east-360*zone), north)
        query = box(*query_bounds)
        for code, edge in dataset.candidate_edges(query_bounds, context=context):
            budget[0] += 1
            if budget[0] > MAX_WORK_ITEMS:
                raise ValueError("State clipping work budget exceeded; split the input for geography analysis")
            _checkpoint(context)
            # Canonicalize orientation before geometric operations: neighboring
            # states usually store the same edge in opposite directions. They
            # must produce one identical station, not a roundoff-sized sliver.
            clipped = LineString(sorted(tuple(p) for p in edge)).intersection(query)
            if clipped.geom_type != "LineString" or clipped.is_empty:
                continue
            a, b = sorted((clipped.coords[0], clipped.coords[-1]))
            a = (a[0]+360*zone, a[1])
            b = (b[0]+360*zone, b[1])
            edge_key = (code, tuple(a), tuple(b))
            if edge_key in seen:
                continue
            seen.add(edge_key)

            if transformer is None:
                transformer = _local_projection(*start, geod)
                # Audit this actual operation, not only its requested CRS.
                # These points lie on the original geodesic at known stations.
                for distance in (0.0, length/2, length):
                    point = start if distance == 0 else source_point(distance)
                    x, y = transformer.transform(*point)
                    radial_error = math.hypot(x-distance*along[0], y-distance*along[1])
                    if not math.isfinite(radial_error) or radial_error > ROOT_DISTANCE_METERS:
                        raise _NumericalClippingError("Local projection failed its source-geodesic accuracy check")

            def projected(t):
                x, y = transformer.transform(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]))
                return x*along[0]+y*along[1], x*normal[0]+y*normal[1]

            samples = [(i/4, projected(i/4)) for i in range(5)]
            coincidence = _shared_stations(start, finish, a, b, length, geod)
            if coincidence is not None:
                if not all(abs(p[1]) <= COINCIDENCE_ROUNDOFF_METERS for _, p in samples):
                    raise _NumericalClippingError("Verified shared geometry failed its projection roundoff check")
                lo, hi = coincidence
                events.extend([lo, hi])
                shared.append((lo, hi, code))
                continue

            dense = [samples[0]]

            def subdivide(t0, p0, t1, p1, depth=0):
                budget[0] += 1
                if budget[0] % 256 == 0:
                    _checkpoint(context)
                if budget[0] > MAX_WORK_ITEMS or depth > 32:
                    raise ValueError("State boundary refinement exceeded its numerical work budget")
                tm = (t0+t1)/2
                pm = projected(tm)
                error = math.hypot(pm[0]-(p0[0]+p1[0])/2, pm[1]-(p0[1]+p1[1])/2)
                if error > PROJECTION_ERROR_METERS:
                    subdivide(t0, p0, tm, pm, depth+1)
                    subdivide(tm, pm, t1, p1, depth+1)
                else:
                    dense.append((t1, p1))

            for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
                subdivide(t0, p0, t1, p1)
            for (t0, p0), (t1, p1) in zip(dense, dense[1:]):
                brackets = [(t0, p0), (t1, p1)]
                # A boundary curve can graze the radial source without changing
                # endpoint signs. Search its interior extremum before ruling it
                # out; this preserves short enter/exit intervals and touches.
                if p0[1]*p1[1] > 0 and min(abs(p0[1]), abs(p1[1])) <= 4*PROJECTION_ERROR_METERS:
                    sign = 1 if p0[1] > 0 else -1
                    optimum = minimize_scalar(lambda t: sign*projected(t)[1], bounds=(t0, t1), method="bounded",
                                              options={"xatol": 1e-14})
                    brackets.insert(1, (float(optimum.x), projected(float(optimum.x))))
                for (left, pl), (right, pr) in zip(brackets, brackets[1:]):
                    candidates = []
                    if pl[1] == 0:
                        candidates.append(left)
                    if pr[1] == 0:
                        candidates.append(right)
                    if pl[1]*pr[1] < 0:
                        candidates.append(brentq(lambda t: projected(t)[1], left, right, xtol=5e-15, rtol=1e-14))
                    for candidate in candidates:
                        distance, residual = projected(candidate)
                        if abs(residual) > ROOT_DISTANCE_METERS:
                            raise ValueError("Boundary crossing root failed numerical verification")
                        if 0.0 < distance < length:
                            events.append(distance)

    # Proven source endpoints already use their canonical stations. Distinct
    # events remain distinct; there is no short-piece threshold or balancing.
    stations = sorted(set(events))
    intervals = []
    for lo, hi in zip(stations, stations[1:]):
        middle = (lo+hi)/2
        boundary_codes = sorted({code for a, b, code in shared if a <= middle <= b})
        lon, lat, _ = geod.fwd(*start, azimuth, middle)
        codes = dataset.states_at(lon, lat)
        if len(boundary_codes) >= 2:
            kind, codes = "shared", boundary_codes
        elif len(codes) == 1:
            kind = "state"
        elif not codes and len(boundary_codes) == 1:
            kind, codes = "state", boundary_codes
        elif not codes:
            kind = "outside"
        else:
            # Multiple memberships alone cannot distinguish polygon overlap
            # from a boundary event whose ownership is numerically uncertain.
            kind = "unresolved"
        start_coord = list(start) if lo == 0 else list(geod.fwd(*start, azimuth, lo)[:2])
        end_coord = list(finish) if hi == length else list(geod.fwd(*start, azimuth, hi)[:2])
        intervals.append((lo, hi, kind, codes, start_coord, end_coord))
    return intervals


def partition_pipelines(pipelines, geod, *, context=None, boundaries=None):
    dataset = boundaries if boundaries is not None else load_boundaries()
    total_edges = sum(len(path)-1 for pipeline in pipelines
                      for path in coordinate_paths_for_pipeline(pipeline, context=context))
    completed_edges = 0
    fragments = []
    diagnostics = []
    source_lengths = {}
    budget = [0]
    failed = False
    crossing_count = 0
    for source_index, pipeline in enumerate(pipelines):
        _checkpoint(context)
        if context is not None:
            context.report("Splitting geometry", completed_edges, total_edges)
        source_id = int(pipeline.get("id", source_index))
        if source_id in source_lengths:
            raise ValueError("Duplicate source identity cannot be partitioned unambiguously")
        source_total = 0.0
        for path_index, path in enumerate(coordinate_paths_for_pipeline(pipeline, context=context)):
            station = 0.0
            previous_state_codes = None
            for original_start, original_finish in zip(path, path[1:]):
                _checkpoint(context)
                if context is not None and completed_edges % 256 == 0:
                    context.report("Splitting geometry", completed_edges, total_edges)
                completed_edges += 1
                azimuth, _, distance = geod.inv(*original_start, *original_finish)
                if not math.isfinite(distance):
                    raise ValueError("Non-finite source geodesic cannot be partitioned")
                distance = abs(distance)
                if distance == 0:
                    continue
                chunks = max(1, math.ceil(distance/MAX_CHUNK_METERS))
                for index in range(chunks):
                    budget[0] += 1
                    lo = distance*index/chunks
                    hi = distance*(index+1)/chunks
                    start = original_start if index == 0 else geod.fwd(*original_start, azimuth, lo)[:2]
                    finish = original_finish if index == chunks-1 else geod.fwd(*original_start, azimuth, hi)[:2]
                    local_azimuth, _, _ = geod.inv(*start, *finish)
                    try:
                        if failed:
                            raise ValueError("Clipping stopped after an earlier numerical or work-limit failure")
                        if budget[0] > MAX_WORK_ITEMS:
                            raise ValueError("Source geometry exceeded the state clipping work budget")
                        intervals = _edge_events(start, finish, local_azimuth, hi-lo, dataset, geod, context, budget)
                    except AnalysisCancelled:
                        raise
                    except Exception as error:
                        if not failed:
                            diagnostic_code = ("state_clipping_numerical_uncertainty" if isinstance(error, _NumericalClippingError)
                                               else "state_clipping_incomplete")
                            diagnostics.append({"level": "error", "code": diagnostic_code,
                                                "message": "Some source geometry could not be assigned reliably and remains unresolved.",
                                                "context": {"error": str(error), "source_id": source_id, "path_index": path_index}})
                        failed = True
                        intervals = [(0.0, hi-lo, "unresolved", [], list(start), list(finish))]
                    for a, b, kind, codes, first, last in intervals:
                        absolute_start, absolute_end = station+lo+a, station+lo+b
                        if b <= a:
                            continue
                        if kind == "state":
                            if previous_state_codes is not None and previous_state_codes != codes:
                                crossing_count += 1
                            previous_state_codes = codes
                        elif kind != "shared":
                            # A shared run can connect two exclusive states, but
                            # outside/unresolved intervals cannot prove a crossing.
                            previous_state_codes = None
                        current = {"id": f"f{len(fragments)}", "source_id": source_id,
                                   "source_name": pipeline.get("name", ""), "placemark_id": pipeline.get("placemark_id"),
                                   "objectid": pipeline.get("objectid", "N/A"), "source_kml": pipeline.get("source_kml", ""),
                                   "path_index": path_index, "start_m": absolute_start, "end_m": absolute_end,
                                   "length_meters": b-a, "coordinates": [first, last], "kind": kind, "state_codes": codes}
                        prev = fragments[-1] if fragments else None
                        can_merge = (prev is not None and prev["source_id"] == source_id and prev["path_index"] == path_index
                                     and prev["kind"] == kind and prev["state_codes"] == codes
                                     # Iteration already proves consecutive
                                     # positive intervals of this source path.
                                     # Recomputed stations may differ by an ULP.
                                     and abs(prev["coordinates"][-1][0]-first[0]) < 180)
                        if can_merge:
                            prev["end_m"] = absolute_end
                            prev["length_meters"] += b-a
                            # An endpoint can also be encountered by another
                            # boundary edge. Retain its coordinate once; this
                            # equality check never removes a distinct crossing.
                            if prev["coordinates"][-1] != last:
                                prev["coordinates"].append(last)
                        else:
                            fragments.append(current)
                station += distance
                source_total += distance
        source_lengths[source_id] = source_total

    by_source = defaultdict(list)
    for fragment in fragments:
        by_source[fragment["source_id"]].append(fragment)
    per_source = []
    for source_id, original in source_lengths.items():
        _checkpoint(context)
        source_fragments = by_source[source_id]
        actual = math.fsum(f["length_meters"] for f in source_fragments)
        # Independent coordinates audit prevents a conserved ledger from masking
        # misplaced cut points or accidental connections in exported geometry.
        physical = math.fsum(abs(geod.inv(*a, *b)[2]) for f in source_fragments
                             for a, b in zip(f["coordinates"], f["coordinates"][1:]))
        tolerance = max(0.001, original*1e-10)
        per_source.append({"source_id": source_id, "original_meters": original, "partitioned_meters": actual,
                           "geometry_meters": physical, "difference_meters": actual-original,
                           "geometry_difference_meters": physical-original, "tolerance_meters": tolerance,
                           "passed": abs(actual-original) <= tolerance and abs(physical-original) <= tolerance})
    original_total = math.fsum(source_lengths.values())
    partitioned_total = math.fsum(f["length_meters"] for f in fragments)
    tolerance = max(0.001, original_total*1e-10)
    passed = all(row["passed"] for row in per_source) and abs(partitioned_total-original_total) <= tolerance
    if not passed:
        diagnostics.append({"level": "error", "code": "state_reconciliation_failed",
                            "message": "State geometry failed the source-mileage conservation audit. No balancing adjustment was made."})
    if any(f["kind"] == "unresolved" for f in fragments) and not any(d.get("level") == "error" for d in diagnostics):
        diagnostics.append({"level": "error", "code": "ambiguous_state_coverage",
                            "message": "State membership could not be uniquely verified; the affected geometry remains unresolved."})
    _checkpoint(context)
    if context is not None:
        context.report("Splitting geometry", total_edges, total_edges)
    return {"boundary_source": dataset.boundary_source, "state_names": dataset.state_names,
            "fragments": fragments, "diagnostics": diagnostics, "crossing_count": crossing_count,
            "reconciliation": {"original_meters": original_total, "partitioned_meters": partitioned_total,
                               "difference_meters": partitioned_total-original_total, "tolerance_meters": tolerance,
                               "passed": passed, "per_source": per_source,
                               "numerical_target_meters": 0.01}}
