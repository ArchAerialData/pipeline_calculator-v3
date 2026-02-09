from __future__ import annotations


def segment_pipeline(geod, coordinates, segment_length):
    """Break a pipeline polyline into fixed-length segments.

    Args:
      geod: pyproj.Geod (or compatible) used for geodesic distance/bearing.
      coordinates: list of (lon, lat)
      segment_length: segment length in meters (float)
    """
    segments = []

    if len(coordinates) < 2:
        return segments

    accumulated_distance = 0.0

    try:
        for i in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]

            azimuth, _, distance = geod.inv(lon1, lat1, lon2, lat2)
            accumulated_distance += distance

            while accumulated_distance >= segment_length:
                ratio = (segment_length - (accumulated_distance - distance)) / distance
                mid_lon = lon1 + ratio * (lon2 - lon1)
                mid_lat = lat1 + ratio * (lat2 - lat1)

                segments.append(
                    {
                        "midpoint": (mid_lon, mid_lat),
                        "bearing": azimuth,
                        "length": segment_length,
                        "segment_index": len(segments),
                    }
                )

                accumulated_distance -= segment_length
                lon1, lat1 = mid_lon, mid_lat
    except Exception as e:
        print(f"Warning: Error segmenting pipeline: {str(e)}")

    return segments

