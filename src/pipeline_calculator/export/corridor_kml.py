from __future__ import annotations

from xml.sax.saxutils import escape as _xml_escape
from xml.etree import ElementTree as ET
from pipeline_calculator.export.geometry_validation import prepare_geometry
from pipeline_calculator.core.corridor_geometry import has_canonical_geometry, normalize_polygons, prepare_corridor
from pipeline_calculator.export.corridor_metadata import validate_corridor_results


def build_overlap_corridor_kml(section, index):
    """Build a KML document for a bundled corridor section.

    Separated from GUI side-effects (tempfile + open) so it can be unit-tested
    and reused by both the legacy monolith and the refactored package modules.
    """
    if has_canonical_geometry(section):
        from pipeline_calculator.export.geography_kmz import append_corridor, document
        validate_corridor_results({'overlap_analysis': {'bundled_sections': [section]}})
        state_scope = 'clipped_polygons' in section or bool(section.get('state_code'))
        prepared = prepare_corridor(section, require_clipped=state_scope)
        if prepared['visualization_status'] == 'omitted':
            shape_scope = 'clipped corridor' if state_scope else 'corridor map'
            raise ValueError(f"No usable {shape_scope} geometry is available. See Diagnostics for details.")
        root, doc = document("State overlap corridor" if state_scope else "Overlap corridor",
                             "Approximate overlap geometry clipped to the selected state." if state_scope else
                             "Approximate overlap area. Polygon geometry does not affect pipeline mileage.")
        append_corridor(doc, prepared, index, require_clipped=state_scope)
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

