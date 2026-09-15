from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape
from xml.etree import ElementTree as ET
from pipeline_calculator.export.geometry_validation import prepare_geometry
from pipeline_calculator.core.corridor_geometry import normalize_polygons, prepare_corridor


def build_overlap_corridor_kml(section, index):
    """Build a KML document for a bundled corridor section.

    Separated from GUI side-effects (tempfile + open) so it can be unit-tested
    and reused by both the legacy monolith and the refactored package modules.
    """
    if "clipped_polygons" in section:
        from pipeline_calculator.export.geography_kmz import append_corridor, document
        prepared = prepare_corridor(section, require_clipped=True)
        if prepared['visualization_status'] == 'omitted':
            raise ValueError("No usable clipped corridor geometry is available")
        root, doc = document("State overlap corridor", "Approximate overlap geometry clipped to the selected state.")
        append_corridor(doc, prepared, index, require_clipped=True)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    coords_list, center, geometry_kind, approximation = prepare_geometry(section)

    label = (
        f"{section['pipeline_1']} + {section['pipeline_2']} "
        f"({section['bundled_length_miles']:.3f} mi, "
        f"{section['average_separation']:.1f} m)"
    )

    from pipeline_calculator.export.geography_kmz import append_polygons
    geometry = ET.Element('geometry')
    append_polygons(geometry, normalize_polygons([{'outer': coords_list, 'holes': []}]), precision=7)
    polygons_xml = ''.join(ET.tostring(child, encoding='unicode') for child in geometry)

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
      {polygons_xml}
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

