from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape


def build_overlap_corridor_kml(section, index):
    """Build a KML document for a bundled corridor section.

    Separated from GUI side-effects (tempfile + open) so it can be unit-tested
    and reused by both the legacy monolith and the refactored package modules.
    """
    # Prefer a curved corridor polygon if available; next try oriented polygon; otherwise, fall back to bbox rectangle
    coords_list = []  # list of (lon, lat)
    curved = section.get('corridor_polygon')
    oriented = section.get('oriented_polygon')
    if curved and isinstance(curved, (list, tuple)) and len(curved) >= 4:
        try:
            coords_list = [(float(lon), float(lat)) for lon, lat in curved]
            if coords_list[0] != coords_list[-1]:
                coords_list.append(coords_list[0])
        except Exception:
            coords_list = []
    elif oriented and isinstance(oriented, (list, tuple)) and len(oriented) >= 4:
        try:
            coords_list = [(float(lon), float(lat)) for lon, lat in oriented]
            if coords_list[0] != coords_list[-1]:
                coords_list.append(coords_list[0])
        except Exception:
            coords_list = []

    if not coords_list:
        bbox = section.get('bbox')
        if bbox:
            min_lon = bbox['min_lon']
            max_lon = bbox['max_lon']
            min_lat = bbox['min_lat']
            max_lat = bbox['max_lat']
        else:
            lon = section.get('center_lon', 0)
            lat = section.get('center_lat', 0)
            buffer = 0.001
            min_lon = lon - buffer
            max_lon = lon + buffer
            min_lat = lat - buffer
            max_lat = lat + buffer

        coords_list = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
            (min_lon, min_lat),
        ]

    label = (
        f"{section['pipeline_1']} + {section['pipeline_2']} "
        f"({section['bundled_length_miles']:.3f} mi, "
        f"{section['average_separation']:.1f} m)"
    )

    coords_str = "\n              ".join(
        f"{lon:.7f},{lat:.7f},0" for lon, lat in coords_list
    )

    width_text = ''
    try:
        if 'oriented_width_m' in section:
            width_text = f", approx width: {float(section['oriented_width_m']):.1f} m"
    except Exception:
        width_text = ''

    label_xml = _xml_escape(label)
    desc_xml = _xml_escape(
        f"Bundled pipeline survey corridor: {section['bundled_length_miles']:.3f} miles at {section['average_separation']:.1f}m average separation{width_text}"
    )

    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Style id="surveyCorridorStyle">
      <PolyStyle>
        <color>7F00FF00</color>
        <outline>1</outline>
      </PolyStyle>
      <LineStyle>
        <color>FF00FF00</color>
        <width>2</width>
      </LineStyle>
    </Style>
    <Placemark>
      <name>{label_xml}</name>
      <description>{desc_xml}</description>
      <styleUrl>#surveyCorridorStyle</styleUrl>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {coords_str}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
    <Placemark>
      <name>Center: {label_xml}</name>
      <Point>
        <coordinates>{float(section.get('center_lon', coords_list[0][0])):.7f},{float(section.get('center_lat', coords_list[0][1])):.7f},0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>'''

    return kml

