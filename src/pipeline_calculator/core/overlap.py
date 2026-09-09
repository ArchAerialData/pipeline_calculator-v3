from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree

from pipeline_calculator.core.angles import bearing_orientation_diff
from pipeline_calculator.core.coordinates import segment_pipeline_paths
from pipeline_calculator.core.spatial import lonlat_array_to_ecef, compute_origin
from pipeline_calculator.core.bundling import qualifying_sections, savings_from_sections

MAX_ANALYSIS_SEGMENTS = 1_000_000
MAX_CANDIDATE_CHECKS = 5_000_000


def find_parallel_segments(pipelines, geod, segment_length, detection_range, angular_tolerance, progress_callback=None):
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
        if progress_callback:
            progress = 0.5 + (p_idx / max(len(pipelines), 1)) * 0.25  # 50-75% progress
            progress_callback(progress)
        pipeline["segments"] = segment_pipeline_paths(geod, pipeline, segment_length)
        segment_count += len(pipeline["segments"])
        if segment_count > MAX_ANALYSIS_SEGMENTS:
            raise ValueError("Analysis segment limit exceeded; split the dataset or increase segment length")

    # Build spatial index
    all_segments = []
    segment_to_pipeline = {}

    for p_idx, pipeline in enumerate(pipelines):
        for seg in pipeline["segments"]:
            seg_idx = len(all_segments)
            all_segments.append(seg["midpoint"])
            segment_to_pipeline[seg_idx] = (p_idx, seg)

    if not all_segments:
        return {}

    try:
        xy = lonlat_array_to_ecef(all_segments, geod)
        tree = KDTree(xy)
    except Exception as e:
        raise ValueError("Could not build overlap spatial index") from e

    parallel_groups = defaultdict(list)
    seen_segment_pairs = set()
    candidate_checks = 0

    for seg_idx, (p_idx, segment) in segment_to_pipeline.items():
        try:
            nearby_indices = tree.query_ball_point(
                xy[seg_idx], math.hypot(float(detection_range), float(segment_length)) + 1e-8
            )
            candidate_checks += len(nearby_indices)
            if candidate_checks > MAX_CANDIDATE_CHECKS:
                raise ValueError("Neighbor-search limit exceeded; split the dataset or reduce detection range")

            for near_idx in nearby_indices:
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

    sections = qualifying_sections(pipelines, parallel_groups, segment_length, min_parallel_length)
    for qualified in sections:
        p1_idx, p2_idx = qualified["pair"]
        section = qualified["representatives"]
        try:
            bundled_length = qualified["length"]
            avg_distance = np.mean([s["distance"] for s in section])

            all_points = []
            pair_midpoints = []
            for seg in section:
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
            for mid1, mid2 in pair_midpoints:
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
            for x, y in cl_xy:
                t = x * u[0] + y * u[1]
                s = x * v[0] + y * v[1]
                t_vals.append(t)
                s_vals.append(s)
            if not t_vals:
                t_vals = [0.0, bundled_length]
                s_vals = [0.0, 0.0]

            t_min = float(min(t_vals))
            t_max = float(max(t_vals))
            s_mean = float(np.mean(s_vals))

            max_sep_m = 0.0
            for mid1, mid2 in pair_midpoints:
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
            t1 = t_min - pad_m
            t2 = t_max + pad_m
            half_w = width_m / 2.0

            A = (u[0] * t1 + v[0] * (s_mean - half_w), u[1] * t1 + v[1] * (s_mean - half_w))
            B = (u[0] * t2 + v[0] * (s_mean - half_w), u[1] * t2 + v[1] * (s_mean - half_w))
            C = (u[0] * t2 + v[0] * (s_mean + half_w), u[1] * t2 + v[1] * (s_mean + half_w))
            D = (u[0] * t1 + v[0] * (s_mean + half_w), u[1] * t1 + v[1] * (s_mean + half_w))

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
                    p0 = cl_xy[i0]
                    n0 = norms[0]
                    left_pts.append((p0[0] + n0[0] * half_w, p0[1] + n0[1] * half_w))
                    right_pts.append((p0[0] - n0[0] * half_w, p0[1] - n0[1] * half_w))

                    for k in range(1, len(dirs)):
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
                    pend = cl_xy[i_last]
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
                                x1, y1 = seq[i]
                                x2, y2 = seq[i + 1]
                                dists.append(math.hypot(x2 - x1, y2 - y1))
                            if not dists:
                                return False
                            med = float(np.median(dists))
                            return med > 0.5 * width_m and med < 3.0 * width_m
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
                }
            )
        except Exception as e:
            raise ValueError(f"Could not construct bundled section for pipelines {p1_idx}, {p2_idx}") from e

    results["bundled_sections"].sort(key=lambda s: s["bundled_length_miles"], reverse=True)

    for p_idx, pipeline in enumerate(pipelines):
        bundled_count = len(bundled_segments[p_idx])
        bundled_length = bundled_count * segment_length

        results["pipeline_overlaps"][pipeline["name"]] = {
            "bundled_segments": bundled_count,
            "bundled_length_meters": bundled_length,
            "bundled_length_miles": bundled_length / survey_mile_m,
        }

    total_bundled = sum(section["bundled_length_meters"] for section in results["bundled_sections"])
    results["total_bundled_length"] = total_bundled
    results["savings_meters"] = savings_from_sections(pipelines, sections, segment_length)

    if progress_callback:
        progress_callback(1.0)

    return results
