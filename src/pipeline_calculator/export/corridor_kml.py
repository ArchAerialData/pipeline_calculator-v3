from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape
from pipeline_calculator.export.geometry_validation import prepare_geometry


def build_overlap_corridor_kml(section, index):
    """Build a KML document for a bundled corridor section.

    Separated from GUI side-effects (tempfile + open) so it can be unit-tested
    and reused by both the legacy monolith and the refactored package modules.
    """
    coords_list, center, geometry_kind, approximation = prepare_geometry(section)

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
        if 'oriented_width_m' in section and geometry_kind == 'sampled_curve':
            width_text = f", approx width: {float(section['oriented_width_m']):.1f} m"
    except Exception:
        width_text = ''

    label_xml = _xml_escape(label)
    desc_xml = _xml_escape(
        f"Approximate bundled pipeline area: {section['bundled_length_miles']:.3f} miles at {section['average_separation']:.1f}m average separation{width_text}. Geometry: {geometry_kind}. {approximation}"
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
        <coordinates>{center[0]:.7f},{center[1]:.7f},0</coordinates>
      </Point>
    </Placemark>
  </Document>
</kml>'''

    return kml

