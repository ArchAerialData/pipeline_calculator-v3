"""KML maps from the canonical partition ledger, with no duplicate line overlays."""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

from pipeline_calculator.core.constants import SURVEY_MILE_METERS
from pipeline_calculator.core.corridor_geometry import GEOD, prepare_corridor, prepare_scope_visualizations

KML_NS = "http://www.opengis.net/kml/2.2"
ET.register_namespace("", KML_NS)


def _tag(name):
    return f"{{{KML_NS}}}{name}"


def element(parent, tag, text=None, **attributes):
    child = ET.SubElement(parent, _tag(tag), attributes)
    if text is not None:
        child.text = str(text)
    return child


def document(name, description):
    root = ET.Element(_tag("kml"))
    doc = element(root, "Document")
    element(doc, "name", name)
    element(doc, "description", description)
    style = element(doc, "Style", id="corridor")
    element(element(style, "PolyStyle"), "color", "6500CC55")
    line = element(style, "LineStyle")
    element(line, "color", "FF009944")
    element(line, "width", "2")
    return root, doc


def _coordinates(points, *, ring=False, precision=None):
    cleaned = []
    for point in points:
        lon, lat = map(float, point[:2])
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Map geometry contains non-finite or out-of-range coordinates")
        if not cleaned or (lon, lat) != cleaned[-1]:
            cleaned.append((lon, lat))
    if ring and cleaned and cleaned[0] != cleaned[-1]:
        cleaned.append(cleaned[0])
    if len(cleaned) < (4 if ring else 2):
        raise ValueError("Map geometry has too few distinct coordinates")
    # Preserve sub-centimeter crossings rather than rounding to display precision.
    if precision is not None:
        return ' '.join(f'{lon:.{precision}f},{lat:.{precision}f},0' for lon, lat in cleaned)
    # repr preserves every input float, including a certified state boundary edge.
    return " ".join(f"{lon!r},{lat!r},0" for lon, lat in cleaned)


def append_polygons(parent, polygons, *, precision=None):
    """Serialize verified clipped polygons, preserving multipart components and holes."""
    from shapely.geometry import Polygon

    if not polygons:
        raise ValueError("No usable clipped corridor geometry is available")
    container = element(parent, "MultiGeometry") if len(polygons) > 1 else parent
    for polygon in polygons:
        outer, holes = polygon["outer"], polygon.get("holes", [])
        if not Polygon(outer, holes).is_valid:
            raise ValueError("Clipped corridor polygon is invalid")
        shape = element(container, "Polygon")
        ring = element(element(shape, "outerBoundaryIs"), "LinearRing")
        element(ring, "coordinates", _coordinates(outer, ring=True, precision=precision))
        for hole in holes:
            ring = element(element(shape, "innerBoundaryIs"), "LinearRing")
            element(ring, "coordinates", _coordinates(hole, ring=True, precision=precision))


def corridor_polygons(section, *, require_clipped=False):
    return prepare_corridor(section, require_clipped=require_clipped)['visualization_polygons']


def append_corridor(parent, section, index, *, require_clipped=False):
    prepared = prepare_corridor(section, require_clipped=require_clipped)
    polygons = prepared['visualization_polygons']
    if not polygons:
        return
    placemark = element(parent, "Placemark")
    element(placemark, "name", f"Corridor {index}: {section.get('pipeline_1', '')} + {section.get('pipeline_2', '')}")
    element(placemark, "description", "Approximate overlap area; polygon geometry is not included in line mileage. "
            + prepared.get('visualization_approximation', ''))
    element(placemark, "styleUrl", "#corridor")
    source_ids = {key: section[key] for key in ("pipeline_1_id", "pipeline_2_id") if key in section}
    if source_ids:
        metadata = element(placemark, "ExtendedData")
        for key, value in source_ids.items():
            element(element(metadata, "Data", name=key), "value", value)
    append_polygons(placemark, polygons)


def _append_fragment(parent, fragment):
    placemark = element(parent, "Placemark")
    element(placemark, "name", fragment.get("source_name", ""))
    miles = fragment["length_meters"] / SURVEY_MILE_METERS
    description = f"{fragment['kind'].title()} geometry: {miles:.9f} US survey miles."
    if fragment["kind"] == "shared":
        description += (f" Shared by {', '.join(fragment['state_codes'])}; each state is assigned "
                        f"{miles / len(fragment['state_codes']):.9f} mi. State overlap: Not calculated.")
    element(placemark, "description", description)
    metadata = element(placemark, "ExtendedData")
    for key in ("id", "source_id", "placemark_id", "objectid", "source_kml", "path_index", "start_m", "end_m", "length_meters"):
        element(element(metadata, "Data", name=key), "value", fragment.get(key, ""))
    paths = _map_line_paths(fragment['coordinates'])
    container = element(placemark, 'MultiGeometry') if len(paths) > 1 else placemark
    for path in paths:
        line = element(container, "LineString")
        element(line, "tessellate", "1")
        element(line, "coordinates", _coordinates(path))


def _map_line_paths(coordinates):
    """Split dateline map edges on their original GRS80 geodesic, without bridges."""
    paths = [[coordinates[0]]]
    for b in coordinates[1:]:
        a = paths[-1][-1]
        if abs(a[0]-b[0]) <= 180:
            paths[-1].append(b)
            continue
        # Equal +/-180 endpoints denote the same meridian; keep one longitude zone.
        if abs(a[0]) == 180:
            if len(paths[-1]) == 1:
                paths[-1][0] = [-a[0], a[1]]
            else:
                paths.append([[-a[0], a[1]]])
            paths[-1].append(b)
            continue
        if abs(b[0]) == 180:
            paths[-1].append([-b[0], b[1]])
            continue
        bearing, _, length = GEOD.inv(*a, *b)
        target = 180.0 if a[0] > 0 else -180.0
        lower, upper = 0.0, length
        # Longitude is monotone along this non-polar geodesic in its unwrapped zone.
        for _ in range(60):
            station = (lower+upper)/2
            lon, lat, _ = GEOD.fwd(*a, bearing, station)
            unwrapped = a[0] + (lon-a[0]+180) % 360 - 180
            if (unwrapped < target) == (target > 0):
                lower = station
            else:
                upper = station
            if upper-lower <= 1e-7:
                break
        paths[-1].append([target, lat])
        paths.append([[-target, lat], b])
    return [path for path in paths if len(path) >= 2]


def build_geography_kml(results, state_code=None):
    """A combined map contains each fragment once; a state map has only interiors."""
    geography = results["geography"]
    states = {state["state_code"]: state for state in geography.get("states", [])}
    fragments = geography.get("fragments", [])
    if state_code is not None:
        state = states[state_code]
        fragments = [fragment for fragment in fragments
                     if fragment["kind"] == "state" and state_code in fragment["state_codes"]]
        if not fragments:
            raise ValueError("A state without exclusive interior geometry has no standalone map")
        name = f"{state['state_name']} pipeline analysis"
        description = (
            "Physical line mileage equals exclusive interior mileage. Shared-border allocations are "
            "documented in analysis.xlsx; shared geometry appears once in Combined/analysis.kmz. "
            "Overlap polygons are clipped to this state."
        )
        root, doc = document(name, description)
        interiors = element(doc, "Folder")
        element(interiors, "name", "State Interiors")
        for fragment in fragments:
            _append_fragment(interiors, fragment)
        scope = prepare_scope_visualizations(state, state_code=state_code)
    else:
        root, doc = document("Combined pipeline analysis", (
            "Original source line geometry is partitioned without duplicate overlays. Shared-border "
            "geometry is stored once and allocated equally among adjoining states. See analysis.xlsx "
            "for provenance, coverage exceptions, reconciliation and independent state overlap results."
        ))
        folders = {}
        state_folders = {}
        names = {"state": "State Interiors", "shared": "Shared Borders",
                 "outside": "Outside Coverage", "unresolved": "Unresolved Geometry"}
        for fragment in fragments:
            kind = fragment["kind"]
            if kind not in folders:
                folders[kind] = element(doc, "Folder")
                element(folders[kind], "name", names[kind])
            parent = folders[kind]
            if kind == "state":
                code = fragment["state_codes"][0]
                if code not in state_folders:
                    state_folders[code] = element(parent, "Folder")
                    element(state_folders[code], "name", states.get(code, {}).get("state_name", code))
                parent = state_folders[code]
            _append_fragment(parent, fragment)
        scope = prepare_scope_visualizations(results)
    sections = (scope.get('overlap_analysis') or {}).get('bundled_sections', [])
    omitted = sum(section.get('visualization_status') == 'omitted' for section in sections)
    if omitted:
        description = doc.find(_tag('description'))
        description.text += f' {omitted} corridor visualization(s) omitted; see report diagnostics.'
    if any(section['visualization_status'] == 'ready' for section in sections):
        corridors = element(doc, "Folder")
        element(corridors, "name", "Overlap Areas")
        for index, section in enumerate(sections, start=1):
            # All expected geometry failures have a structured preflight decision.
            # Unexpected exporter/file errors still abort the atomic package.
            if section['visualization_status'] == 'omitted':
                continue
            append_corridor(corridors, section, index, require_clipped=state_code is not None)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def write_geography_kmz(results, destination, state_code=None):
    text = build_geography_kml(results, state_code)
    with ZipFile(Path(destination), "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("doc.kml", text.encode("utf-8"))
