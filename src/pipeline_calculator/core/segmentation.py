from __future__ import annotations

from pipeline_calculator.core.execution import AnalysisCancelled

import math

MAX_ANALYSIS_SEGMENTS = 1_000_000

def segment_pipeline(geod, coordinates, segment_length, *, context=None, max_segments=None):
    """Break a pipeline polyline into fixed-length analysis segments.

    Args:
      geod: pyproj.Geod (or compatible) used for geodesic distance/bearing.
      coordinates: list of (lon, lat)
      segment_length: segment length in meters (float)
    """
    if context is not None:
        context.check()
    segments: list[dict] = []
    limit = MAX_ANALYSIS_SEGMENTS if max_segments is None else min(max_segments, MAX_ANALYSIS_SEGMENTS)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError('Segment budget must be a nonnegative integer')

    try:
        seg_len = float(segment_length)
    except (TypeError, ValueError) as exc:
        raise ValueError("Segment length must be a finite positive number") from exc

    if not math.isfinite(seg_len) or seg_len <= 0:
        raise ValueError("Segment length must be a finite positive number")
    if len(coordinates) < 2:
        return segments

    # We produce N full segments of length `seg_len` along the polyline path and
    # intentionally ignore any trailing remainder (< seg_len). Downstream code
    # accounts for this via a "tails" correction (see effective_length.py).
    #
    # Implementation notes:
    # - The previous implementation attempted to do this via an "accumulated"
    #   approach but updated the edge start point without updating the edge
    #   distance, which clustered many segment points and then created large
    #   gaps on long edges. This version is correct for arbitrarily long KML
    #   edges.
    # - We compute geodesic points along each vertex-to-vertex edge using
    #   `geod.fwd` from the original edge start with the edge's initial forward
    #   azimuth, avoiding an expensive `geod.inv` call per generated segment.
    #
    # Each segment dict includes:
    # - midpoint: (lon, lat) point representing the segment (used for indexing)
    # - bearing: segment forward azimuth in degrees (used for "parallel" checks)
    # - length: segment length in meters (always `seg_len`)
    # - segment_index: 0-based sequential index within the pipeline
    carry_m = 0.0  # meters from last segment boundary to the current vertex
    prev_boundary = tuple(coordinates[0])

    try:
        for i in range(len(coordinates) - 1):
            if context is not None and i % 256 == 0:
                context.check()
            lon_a, lat_a = coordinates[i]
            lon_b, lat_b = coordinates[i + 1]
            if lon_a == lon_b and lat_a == lat_b:
                continue

            # Edge geometry (A -> B).
            az_ab, _, dist_ab = geod.inv(lon_a, lat_a, lon_b, lat_b)
            az_ab = float(az_ab)
            dist_ab = float(dist_ab)
            if not math.isfinite(az_ab) or not math.isfinite(dist_ab):
                raise ValueError("Invalid geodesic segment geometry")
            if dist_ab < 0:
                dist_ab = -dist_ab
                az_ab = az_ab + 180.0

            if dist_ab <= 0:
                continue

            edge_pos_m = 0.0  # distance along the current edge from A
            rem_m = dist_ab
            if carry_m + rem_m + 1e-9 >= (limit - len(segments) + 1) * seg_len:
                raise ValueError("Analysis segment limit exceeded; split the dataset or increase segment length")

            # Generate as many full segments as we can on this edge, accounting
            # for `carry_m` accumulated from previous edges.
            while carry_m + rem_m >= seg_len - 1e-9:
                if len(segments) >= limit:
                    raise ValueError("Analysis segment limit exceeded; split the dataset or increase segment length")
                if context is not None:
                    context.checkpoint()
                if context is not None and len(segments) % 256 == 0:
                    context.report("Segmenting path", len(segments))
                needed_m = seg_len - carry_m
                if needed_m <= 1e-12:
                    # Defensive: if floating error yields ~0, snap to a fresh segment.
                    needed_m = seg_len
                    carry_m = 0.0

                edge_pos_m += needed_m
                if edge_pos_m > dist_ab:
                    edge_pos_m = dist_ab

                # Segment boundary at distance `edge_pos_m` from the vertex A.
                seg_end_lon, seg_end_lat, _ = geod.fwd(lon_a, lat_a, az_ab, edge_pos_m)
                seg_end = (float(seg_end_lon), float(seg_end_lat))

                # Do not substitute invented geometry if geodesic operations fail.
                seg_bearing, _, seg_dist = geod.inv(*prev_boundary, *seg_end)
                seg_bearing, seg_dist = float(seg_bearing), abs(float(seg_dist))
                mid_lon, mid_lat, _ = geod.fwd(*prev_boundary, seg_bearing, seg_dist / 2)
                midpoint = (float(mid_lon), float(mid_lat))
                if not all(math.isfinite(v) for v in (*midpoint, seg_bearing, seg_dist)):
                    raise ValueError("Non-finite segment geometry")

                segments.append(
                    {
                        "midpoint": midpoint,
                        "bearing": seg_bearing,
                        "length": seg_len,
                        "segment_index": len(segments),
                    }
                )

                prev_boundary = seg_end
                rem_m = dist_ab - edge_pos_m
                carry_m = 0.0

                if rem_m <= 1e-9:
                    rem_m = 0.0
                    break

            # Any remaining edge distance (that didn't complete a segment) is
            # carried forward to the next vertex-to-vertex edge.
            carry_m += rem_m
    except AnalysisCancelled:
        raise
    except Exception as e:
        raise ValueError(f"Could not segment pipeline; partial segments were discarded: {e}") from e

    return segments
