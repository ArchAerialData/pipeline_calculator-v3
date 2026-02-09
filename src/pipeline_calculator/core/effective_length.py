from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree

from pipeline_calculator.core.angles import bearing_orientation_diff
from pipeline_calculator.core.segmentation import segment_pipeline
from pipeline_calculator.core.spatial import compute_origin, lonlat_array_to_xy


def compute_effective_length_by_clusters(
    pipelines,
    per_pipeline_total_meters,
    geod,
    segment_length,
    detection_range,
    angular_tolerance,
    progress_callback=None,
):
    """Compute effective length using per-segment clustering across pipelines.

    For each segment midpoint, find nearby parallel segments on other pipelines
    within detection range. If k pipelines share that neighborhood, attribute
    only 1/k of that segment length to the effective total.
    """
    for pipeline in pipelines:
        if "segments" not in pipeline:
            pipeline["segments"] = segment_pipeline(geod, pipeline["coordinates"], segment_length)

    all_midpoints = []
    seg_index_map = {}
    per_pipeline_segment_sum = defaultdict(float)

    for p_idx, pipeline in enumerate(pipelines):
        for seg in pipeline["segments"]:
            g = len(all_midpoints)
            all_midpoints.append(seg["midpoint"])
            seg_index_map[g] = (p_idx, seg)
            per_pipeline_segment_sum[p_idx] += float(seg.get("length", segment_length))

    if not all_midpoints:
        return float(sum(per_pipeline_total_meters))

    try:
        lon0, lat0 = compute_origin(all_midpoints)
        xy = lonlat_array_to_xy(all_midpoints, lon0, lat0)
        tree = KDTree(xy)
    except Exception:
        return float(sum(per_pipeline_total_meters))

    eff_total = 0.0

    for g_idx, (p_idx, seg) in seg_index_map.items():
        try:
            neighbor_ids = tree.query_ball_point(xy[g_idx], float(detection_range))
        except Exception:
            neighbor_ids = []

        participating = {p_idx}

        for n_idx in neighbor_ids:
            if n_idx == g_idx:
                continue
            if n_idx not in seg_index_map:
                continue
            np_idx, nseg = seg_index_map[n_idx]
            if np_idx == p_idx:
                continue

            if bearing_orientation_diff(seg["bearing"], nseg["bearing"]) > angular_tolerance:
                continue

            lon1, lat1 = seg["midpoint"]
            lon2, lat2 = nseg["midpoint"]
            _, _, d_m = geod.inv(lon1, lat1, lon2, lat2)
            if d_m <= detection_range:
                participating.add(np_idx)

        k = max(1, len(participating))
        seg_len = float(seg.get("length", segment_length))
        eff_total += seg_len / k

    tails = 0.0
    for p_idx, tot in enumerate(per_pipeline_total_meters):
        segmented = per_pipeline_segment_sum.get(p_idx, 0.0)
        if tot > segmented:
            tails += (tot - segmented)
    eff_total += tails

    return eff_total
