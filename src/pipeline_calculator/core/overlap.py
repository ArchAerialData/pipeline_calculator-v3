from __future__ import annotations

from pipeline_calculator.core.execution import AnalysisCancelled

import math
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree

from pipeline_calculator.core.angles import bearing_orientation_diff
from pipeline_calculator.core.coordinates import segment_pipeline_paths
from pipeline_calculator.core.spatial import lonlat_array_to_ecef, compute_origin
from pipeline_calculator.core.bundling import qualifying_sections, savings_from_sections
from pipeline_calculator.core.workload import check_density_workload

MAX_ANALYSIS_SEGMENTS = 1_000_000
MAX_CANDIDATE_CHECKS = 5_000_000


def find_parallel_segments(pipelines, geod, segment_length, detection_range, angular_tolerance, progress_callback=None, *, context=None):
    """Identify pipeline segments that run parallel within detection range.

    Mutates pipelines by adding a `segments` list on each pipeline dict.
    """
    if not math.isfinite(float(segment_length)) or segment_length <= 0:
        raise ValueError("Segment length must be a finite positive number")
    if not all(math.isfinite(float(v)) for v in (detection_range, angular_tolerance)):
        raise ValueError("Analysis parameters must be finite")
    if segment_length <= 0 or detection_range <= 0 or not 0 <= angular_tolerance <= 90:
        raise ValueError("Segment length and detection range must be positive; angle must be between 0 and 90")
    # Segment all pipelines
    segment_count = 0
    for p_idx, pipeline in enumerate(pipelines):
        if context is not None and p_idx % 256 == 0:
            context.check()
        if context is not None:
            context.report("Segmenting paths", p_idx, len(pipelines))
        if progress_callback:
            progress = 0.5 + (p_idx / max(len(pipelines), 1)) * 0.25  # 50-75% progress
            progress_callback(progress)
        pipeline["segments"] = segment_pipeline_paths(geod, pipeline, segment_length, context=context)
        segment_count += len(pipeline["segments"])
        if segment_count > MAX_ANALYSIS_SEGMENTS:
            raise ValueError("Analysis segment limit exceeded; split the dataset or increase segment length")

    if context is not None:
        context.report("Building spatial index", 0, segment_count)
    # Build spatial index
    all_segments = []
    segment_to_pipeline = {}

    for p_idx, pipeline in enumerate(pipelines):
        if context is not None and p_idx % 256 == 0:
            context.check()
        for segment_position, seg in enumerate(pipeline['segments']):
            if context is not None and segment_position % 256 == 0:
                context.check()
            seg_idx = len(all_segments)
            all_segments.append(seg["midpoint"])
            segment_to_pipeline[seg_idx] = (p_idx, seg)

    if not all_segments:
        return {}

    try:
        xy = lonlat_array_to_ecef(all_segments, geod)
        tree = KDTree(xy)
    except AnalysisCancelled:
        raise
    except Exception as e:
        raise ValueError("Could not build overlap spatial index") from e

    radius = math.hypot(float(detection_range), float(segment_length)) + 1e-8
    check_density_workload(tree, xy, radius, context)
    if context is not None:
        context.report("Searching neighbors", 0, len(all_segments))
    parallel_groups = defaultdict(list)
    seen_segment_pairs = set()
    candidate_checks = 0

    for seg_idx, (p_idx, segment) in segment_to_pipeline.items():
        if context is not None and seg_idx % 256 == 0:
            context.report("Searching neighbors", seg_idx, len(all_segments))
        try:
            nearby_indices = tree.query_ball_point(
                xy[seg_idx], radius
            )
            candidate_checks += len(nearby_indices)
            if candidate_checks > MAX_CANDIDATE_CHECKS:
                raise ValueError("Neighbor-search limit exceeded; split the dataset or reduce detection range")

            for candidate_position, near_idx in enumerate(nearby_indices):
                if context is not None and candidate_position % 256 == 0:
                    context.check()
                if near_idx == seg_idx:
                    continue
                if near_idx not in segment_to_pipeline:
                    continue

                near_p_idx, near_segment = segment_to_pipeline[near_idx]
                if p_idx == near_p_idx:
                    continue

                if bearing_orientation_diff(segment["bearing"], near_segment["bearing"]) <= angular_tolerance:
                    lon1, lat1 = segment["midpoint"]
                    lon2, lat2 = near_segment["midpoint"]
                    azimuth, _, midpoint_distance = geod.inv(lon1, lat1, lon2, lat2)
                    delta1 = math.radians(azimuth - segment["bearing"])
                    # Back azimuth at the second midpoint handles meridian convergence.
                    reverse_azimuth, _, _ = geod.inv(lon2, lat2, lon1, lat1)
                    delta2 = math.radians(reverse_azimuth - near_segment["bearing"])
                    distance = max(abs(midpoint_distance * math.sin(delta1)),
                                   abs(midpoint_distance * math.sin(delta2)))
                    half_extent = (segment["length"] + near_segment["length"]) / 2
                    along = max(abs(midpoint_distance * math.cos(delta1)),
                                abs(midpoint_distance * math.cos(delta2)))
                    # Compare short, finite segment tangents instead of requiring
                    # their midpoint samples to line up. Touching endpoints alone
                    # do not constitute longitudinal overlap.
                    if distance <= detection_range and along < half_extent - 1e-8:
                        pair_id = (min(seg_idx, near_idx), max(seg_idx, near_idx))
                        if pair_id in seen_segment_pairs:
                            continue
                        seen_segment_pairs.add(pair_id)

                        key = tuple(sorted([p_idx, near_p_idx]))
                        if key[0] == p_idx:
                            parallel_groups[key].append(
                                {
                                    "pipeline_1_segment": segment["segment_index"],
                                    "pipeline_2_segment": near_segment["segment_index"],
                                    "pipeline_1_path": segment.get("path_index", 0),
                                    "pipeline_2_path": near_segment.get("path_index", 0),
                                    "pipeline_1_path_segment": segment.get(
                                        "path_segment_index",
                                        segment["segment_index"],
                                    ),
                                    "pipeline_2_path_segment": near_segment.get(
                                        "path_segment_index",
                                        near_segment["segment_index"],
                                    ),
                                    "distance": distance,
                                    "midpoint_distance": midpoint_distance,
                                }
                            )
                        else:
                            parallel_groups[key].append(
                                {
                                    "pipeline_1_segment": near_segment["segment_index"],
                                    "pipeline_2_segment": segment["segment_index"],
                                    "pipeline_1_path": near_segment.get("path_index", 0),
                                    "pipeline_2_path": segment.get("path_index", 0),
                                    "pipeline_1_path_segment": near_segment.get(
                                        "path_segment_index",
                                        near_segment["segment_index"],
                                    ),
                                    "pipeline_2_path_segment": segment.get(
                                        "path_segment_index",
                                        segment["segment_index"],
                                    ),
                                    "distance": distance,
                                    "midpoint_distance": midpoint_distance,
                                }
                            )
        except AnalysisCancelled:
            raise
        except Exception as e:
            raise ValueError(f"Could not analyze segment {seg_idx}: {e}") from e

    return parallel_groups


def calculate_overlap_results(
    pipelines,
    parallel_groups,
    geod,
    survey_mile_m,
    segment_length,
    min_parallel_length,
    detection_range,
    angular_tolerance,
    progress_callback=None,
    *, context=None,
):
    """Calculate bundled lengths and overlap statistics."""
    if not math.isfinite(float(min_parallel_length)) or min_parallel_length <= 0:
        raise ValueError("Minimum parallel length must be finite and positive")
    results = {
        "bundled_sections": [],
        "pipeline_overlaps": {},
        "total_bundled_length": 0,
        "effective_total_length": 0,
        "savings_meters": 0,
        "savings_miles": 0,
        "savings_percentage": 0,
        "parameter_impacts": {},
    }

    bundled_segments = defaultdict(set)

    sections = qualifying_sections(pipelines, parallel_groups, segment_length, min_parallel_length, context=context)
    if context is not None:
        context.report("Building corridors", 0, len(sections))
    for section_index, qualified in enumerate(sections):
        if context is not None and section_index % 256 == 0:
            context.check()
        if context is not None:
            context.report("Building corridors", section_index, len(sections))
        p1_idx, p2_idx = qualified["pair"]
        section = qualified["representatives"]
        try:
            bundled_length = qualified["length"]
            avg_distance = np.mean([s["distance"] for s in section])

            all_points = []
            pair_midpoints = []
            for match_position, seg in enumerate(section):
                if context is not None and match_position % 256 == 0:
                    context.check()
                seg1_idx = seg["pipeline_1_segment"]
                seg2_idx = seg["pipeline_2_segment"]

                p1_segments = pipelines[p1_idx]["segments"]
                p2_segments = pipelines[p2_idx]["segments"]

                if seg1_idx >= len(p1_segments) or seg2_idx >= len(p2_segments):
                    print(f"Warning: Invalid segment indices {seg1_idx}, {seg2_idx}")
                    continue

                mid1 = p1_segments[seg1_idx]["midpoint"]
                mid2 = p2_segments[seg2_idx]["midpoint"]
                all_points.extend([mid1, mid2])
                pair_midpoints.append((mid1, mid2))

            # Bounds must include every qualified sample, including partner
            # samples not chosen for the representative centerline.
            all_points = []
            for pipe_index, segment_ids in zip((p1_idx, p2_idx), qualified['segment_ids']):
                for position, segment_id in enumerate(sorted(segment_ids)):
                    if context is not None and position % 256 == 0:
                        context.check()
                    all_points.append(pipelines[pipe_index]['segments'][segment_id]['midpoint'])
            if not all_points:
                continue

            origin_lon, _ = compute_origin(all_points)
            # Unwrap around this corridor, not Greenwich, before finding bounds.
            lons = [origin_lon + (p[0] - origin_lon + 180) % 360 - 180 for p in all_points]
            lats = [p[1] for p in all_points]
            min_lon, max_lon = min(lons) - 0.001, max(lons) + 0.001
            min_lat, max_lat = max(-90, min(lats) - 0.001), min(90, max(lats) + 0.001)
            center_lon = ((min_lon + max_lon) / 2 + 180) % 360 - 180
            center_lat = (min_lat + max_lat) / 2
            # Wrapped west/east bounds may have west > east at the dateline.
            min_lon = (min_lon + 180) % 360 - 180
            max_lon = (max_lon + 180) % 360 - 180

            centerline_pts = []
            for midpoint_position, (mid1, mid2) in enumerate(pair_midpoints):
                if context is not None and midpoint_position % 256 == 0:
                    context.check()
                bearing, _, distance = geod.inv(*mid1, *mid2)
                cl_lon, cl_lat, _ = geod.fwd(*mid1, bearing, distance / 2)
                centerline_pts.append((cl_lon, cl_lat))

            def to_xy(lon, lat):
                bearing, _, distance = geod.inv(center_lon, center_lat, lon, lat)
                radians = math.radians(bearing)
                return distance * math.sin(radians), distance * math.cos(radians)

            def to_lonlat(x, y):
                lon, lat, _ = geod.fwd(center_lon, center_lat,
                                       math.degrees(math.atan2(x, y)), math.hypot(x, y))
                return lon, lat

            cl_xy = [to_xy(lon, lat) for lon, lat in centerline_pts]

            if len(cl_xy) >= 2:
                x0, y0 = cl_xy[0]
                x1, y1 = cl_xy[-1]
                vx, vy = (x1 - x0), (y1 - y0)
                norm = math.hypot(vx, vy)
                if norm < 1e-6:
                    u = (1.0, 0.0)
                else:
                    u = (vx / norm, vy / norm)
            else:
                u = (1.0, 0.0)

            v = (-u[1], u[0])

            t_vals = []
            s_vals = []
            for point_position, (x, y) in enumerate(cl_xy):
                if context is not None and point_position % 256 == 0:
                    context.check()
                t = x * u[0] + y * u[1]
                s = x * v[0] + y * v[1]
                t_vals.append(t)
                s_vals.append(s)
            if not t_vals:
                t_vals = [0.0, bundled_length]
                s_vals = [0.0, 0.0]

            s_mean = float(np.mean(s_vals))

            max_sep_m = 0.0
            for midpoint_position, (mid1, mid2) in enumerate(pair_midpoints):
                if context is not None and midpoint_position % 256 == 0:
                    context.check()
                x1, y1 = to_xy(mid1[0], mid1[1])
                x2, y2 = to_xy(mid2[0], mid2[1])
                sep = math.hypot(x2 - x1, y2 - y1)
                if sep > max_sep_m:
                    max_sep_m = sep

            margin_m = 10.0
            width_m = max(max_sep_m + margin_m, segment_length)
            if detection_range > 0:
                width_m = min(width_m, 2.0 * detection_range)

            pad_m = max(segment_length * 1.0, 5.0)
            # A fallback rectangle encloses a bent group, rather than a narrow
            # strip through its average lateral position. Keep nominal strip
            # width separate from this rectangle's potentially broader extent.
            along, across = [], []
            for position, point in enumerate(all_points):
                if context is not None and position % 256 == 0:
                    context.check()
                x, y = to_xy(*point)
                along.append(x*u[0] + y*u[1])
                across.append(x*v[0] + y*v[1])
            t1 = min(along) - pad_m
            t2 = max(along) + pad_m
            half_w = width_m / 2.0
            s1 = min(min(across) - pad_m, s_mean - half_w)
            s2 = max(max(across) + pad_m, s_mean + half_w)

            A = (u[0] * t1 + v[0] * s1, u[1] * t1 + v[1] * s1)
            B = (u[0] * t2 + v[0] * s1, u[1] * t2 + v[1] * s1)
            C = (u[0] * t2 + v[0] * s2, u[1] * t2 + v[1] * s2)
            D = (u[0] * t1 + v[0] * s2, u[1] * t1 + v[1] * s2)

            oriented_polygon = [
                to_lonlat(A[0], A[1]),
                to_lonlat(B[0], B[1]),
                to_lonlat(C[0], C[1]),
                to_lonlat(D[0], D[1]),
                to_lonlat(A[0], A[1]),
            ]

            def unit(vec):
                vx, vy = vec
                nrm = math.hypot(vx, vy)
                if nrm < 1e-9:
                    return (0.0, 0.0)
                return (vx / nrm, vy / nrm)

            def line_intersection(p, d, q, e):
                cross = d[0] * e[1] - d[1] * e[0]
                if abs(cross) < 1e-9:
                    return None
                r = (q[0] - p[0], q[1] - p[1])
                t = (r[0] * e[1] - r[1] * e[0]) / cross
                return (p[0] + t * d[0], p[1] + t * d[1])

            N = len(cl_xy)
            curved_polygon = None
            if N >= 2:
                dirs = []
                norms = []
                valid_idx = []
                for i in range(N - 1):
                    if context is not None and i % 256 == 0:
                        context.check()
                    dx = cl_xy[i + 1][0] - cl_xy[i][0]
                    dy = cl_xy[i + 1][1] - cl_xy[i][1]
                    udir = unit((dx, dy))
                    if udir == (0.0, 0.0):
                        continue
                    dirs.append(udir)
                    norms.append((-udir[1], udir[0]))
                    valid_idx.append(i)

                if not dirs:
                    curved_polygon = [to_lonlat(px, py) for px, py in [A, B, C, D, A]]
                else:
                    miter_limit = 6.0
                    left_pts = []
                    right_pts = []

                    i0 = valid_idx[0]
                    p0 = (cl_xy[i0][0] - dirs[0][0]*pad_m,
                          cl_xy[i0][1] - dirs[0][1]*pad_m)
                    n0 = norms[0]
                    left_pts.append((p0[0] + n0[0] * half_w, p0[1] + n0[1] * half_w))
                    right_pts.append((p0[0] - n0[0] * half_w, p0[1] - n0[1] * half_w))

                    for k in range(1, len(dirs)):
                        if context is not None and k % 256 == 0:
                            context.check()
                        i_curr = valid_idx[k]
                        pi = cl_xy[i_curr]
                        d_prev = dirs[k - 1]
                        d_curr = dirs[k]
                        n_prev = norms[k - 1]
                        n_curr = norms[k]

                        Lp = (pi[0] + n_prev[0] * half_w, pi[1] + n_prev[1] * half_w)
                        Lc = (pi[0] + n_curr[0] * half_w, pi[1] + n_curr[1] * half_w)
                        left_int = line_intersection(Lp, d_prev, Lc, d_curr)

                        Rp = (pi[0] - n_prev[0] * half_w, pi[1] - n_prev[1] * half_w)
                        Rc = (pi[0] - n_curr[0] * half_w, pi[1] - n_curr[1] * half_w)
                        right_int = line_intersection(Rp, d_prev, Rc, d_curr)

                        def safe_join(prev_pt, cand, curr_pt):
                            if cand is None:
                                return [prev_pt, curr_pt]
                            ml = math.hypot(cand[0] - pi[0], cand[1] - pi[1])
                            if ml > miter_limit * half_w:
                                return [prev_pt, curr_pt]
                            return [cand]

                        left_join = safe_join(Lp, left_int, Lc)
                        right_join = safe_join(Rp, right_int, Rc)

                        left_pts.extend(left_join)
                        right_pts.extend(right_join)

                    i_last = valid_idx[-1] + 1
                    pend = (cl_xy[i_last][0] + dirs[-1][0]*pad_m,
                            cl_xy[i_last][1] + dirs[-1][1]*pad_m)
                    n_last = norms[-1]
                    left_pts.append((pend[0] + n_last[0] * half_w, pend[1] + n_last[1] * half_w))
                    right_pts.append((pend[0] - n_last[0] * half_w, pend[1] - n_last[1] * half_w))

                    ring_xy = list(left_pts) + list(reversed(right_pts))

                    def looks_zigzag(seq):
                        try:
                            sample = min(20, len(seq) - 1)
                            if sample < 4:
                                return False
                            dists = []
                            for i in range(sample):
                                if context is not None and i % 256 == 0:
                                    context.check()
                                x1, y1 = seq[i]
                                x2, y2 = seq[i + 1]
                                dists.append(math.hypot(x2 - x1, y2 - y1))
                            if not dists:
                                return False
                            med = float(np.median(dists))
                            return med > 0.5 * width_m and med < 3.0 * width_m
                        except AnalysisCancelled:
                            raise
                        except Exception:
                            return False

                    if looks_zigzag(ring_xy):
                        curved_polygon = None
                    else:
                        if ring_xy[0] != ring_xy[-1]:
                            ring_xy.append(ring_xy[0])
                        curved_polygon = [to_lonlat(px, py) for (px, py) in ring_xy]

            bundled_segments[p1_idx].update(qualified["segment_ids"][0])
            bundled_segments[p2_idx].update(qualified["segment_ids"][1])

            results["bundled_sections"].append(
                {
                    "pipeline_1": pipelines[p1_idx]["name"],
                    "pipeline_2": pipelines[p2_idx]["name"],
                    "source_path_indices": list(qualified["paths"]),
                    "bundled_length_meters": bundled_length,
                    "bundled_length_miles": bundled_length / survey_mile_m,
                    "average_separation": avg_distance,
                    "segment_count": min(map(len, qualified["segment_ids"])),
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                    "bbox": {
                        "min_lon": min_lon,
                        "max_lon": max_lon,
                        "min_lat": min_lat,
                        "max_lat": max_lat,
                    },
                    "oriented_polygon": oriented_polygon,
                    "oriented_width_m": width_m,
                    "corridor_polygon": curved_polygon if curved_polygon else oriented_polygon,
                    "corridor_geometry_kind": "sampled_curve" if curved_polygon else "oriented_rectangle",
                    "corridor_approximation": "Sampled midpoint visualization; not a surveyed boundary.",
                }
            )
        except AnalysisCancelled:
            raise
        except Exception as e:
            raise ValueError(f"Could not construct bundled section for pipelines {p1_idx}, {p2_idx}") from e

    results["bundled_sections"].sort(key=lambda s: s["bundled_length_miles"], reverse=True)

    for p_idx, pipeline in enumerate(pipelines):
        if context is not None and p_idx % 256 == 0:
            context.check()
        bundled_count = len(bundled_segments[p_idx])
        bundled_length = bundled_count * segment_length

        results["pipeline_overlaps"][pipeline["name"]] = {
            "bundled_segments": bundled_count,
            "bundled_length_meters": bundled_length,
            "bundled_length_miles": bundled_length / survey_mile_m,
        }

    total_bundled = sum(section["bundled_length_meters"] for section in results["bundled_sections"])
    results["total_bundled_length"] = total_bundled
    results["savings_meters"] = savings_from_sections(pipelines, sections, segment_length, context=context)

    if progress_callback:
        progress_callback(1.0)

    return results
