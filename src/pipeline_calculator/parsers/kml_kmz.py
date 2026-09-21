from __future__ import annotations

from pipeline_calculator.core.execution import AnalysisCancelled

from dataclasses import dataclass, field
from collections import deque
import math
from pathlib import Path, PurePosixPath
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse


KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"

# Bound decompressed input and linked-document traversal before parsing XML.
MAX_KML_BYTES = 64 * 1024 * 1024
MAX_TOTAL_KML_BYTES = 256 * 1024 * 1024
MAX_KML_DOCUMENTS = 1024
MAX_ARCHIVE_ENTRIES = 10000
INCOMPLETE_CODES = {
    "linked_kml_parse_error", "unresolved_network_link", "network_link_missing_href",
    "remote_network_link_skipped", "unsupported_network_link_target",
    "network_link_outside_base_skipped", "malformed_placemark",
    "no_supported_features", "short_linestring", "short_gx_track",
    "missing_point_coordinate", "ambiguous_point_coordinate",
}


@dataclass
class ParseResult:
    pipelines: list[dict] = field(default_factory=list)
    placemarks: list[dict] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    parsed_kml_files: list[str] = field(default_factory=list)


@dataclass
class _ParserState:
    context: object = None
    pipelines: list[dict] = field(default_factory=list)
    placemarks: list[dict] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    parsed_kml_files: list[str] = field(default_factory=list)
    pipeline_count: int = 0
    placemark_count: int = 0
    legacy_feature_count: int = 0
    bytes_read: int = 0
    documents_read: int = 0


def _diag(state: _ParserState, code: str, message: str, *, level: str = "warning", **context) -> None:
    if code in INCOMPLETE_CODES:
        level = "error"
    entry = {"level": level, "code": code, "message": message}
    clean_context = {k: v for k, v in context.items() if v not in (None, "", [])}
    if clean_context:
        entry["context"] = clean_context
    state.diagnostics.append(entry)


def _read_document(stream, size, state):
    if size > MAX_KML_BYTES:
        raise ValueError(f"KML document exceeds the {MAX_KML_BYTES // (1024 * 1024)} MiB input limit")
    if state.documents_read >= MAX_KML_DOCUMENTS or state.bytes_read + size > MAX_TOTAL_KML_BYTES:
        raise ValueError("Linked KML input exceeds the document-count or total-size limit")
    if state.context is not None:
        state.context.report("Reading documents", state.documents_read)
    data = stream.read(MAX_KML_BYTES + 1)
    if state.context is not None:
        state.context.check()
    if len(data) > MAX_KML_BYTES or state.bytes_read + len(data) > MAX_TOTAL_KML_BYTES:
        raise ValueError("KML decompressed input exceeds the size limit")
    state.bytes_read += len(data)
    state.documents_read += 1
    return data


def _tag_uri(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") and "}" in tag else ""


def _local_name(tag_or_elem) -> str:
    tag = getattr(tag_or_elem, "tag", tag_or_elem)
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _matches(elem, local_name: str, namespace: str | None = None) -> bool:
    if _local_name(elem) != local_name:
        return False
    return namespace is None or _tag_uri(elem.tag) == namespace


def _iter_desc(elem, local_name: str, namespace: str | None = None, *, context=None):
    for child in elem.iter():
        if context is not None:
            context.checkpoint()
        if _matches(child, local_name, namespace):
            yield child


def _find_child(elem, local_name: str, namespace: str | None = None):
    for child in list(elem):
        if _matches(child, local_name, namespace):
            return child
    return None


def _text(elem) -> str:
    return (elem.text or "").strip() if elem is not None else ""


def _attr(elem, name: str) -> str | None:
    if elem is None:
        return None
    if name in elem.attrib:
        return elem.attrib[name]
    for key, value in elem.attrib.items():
        if _local_name(key) == name:
            return value
    return None


def _parse_coordinates_text(coords_text: str, state: _ParserState, *, source: str, feature_name: str, geometry: str):
    coords = []
    invalid_count = 0

    for coord_str in (coords_text or "").replace("\n", " ").replace("\t", " ").split():
        if state.context is not None:
            state.context.checkpoint()
        try:
            parts = coord_str.split(",")
            if len(parts) < 2:
                invalid_count += 1
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            if -180 <= lon <= 180 and -90 <= lat <= 90:
                coords.append((lon, lat))
            else:
                invalid_count += 1
        except (ValueError, IndexError):
            invalid_count += 1

    if invalid_count:
        _diag(
            state,
            "invalid_coordinate",
            f"Rejected {geometry} containing {invalid_count} invalid coordinate tuple(s); no connections were inferred across missing vertices.",
            level="error",
            source=source,
            feature_name=feature_name,
            geometry=geometry,
        )

        return []

    return coords


def _parse_gx_coord_text(coord_text: str):
    parts = (coord_text or "").strip().split()
    if len(parts) < 2:
        return None
    lon = float(parts[0])
    lat = float(parts[1])
    if -180 <= lon <= 180 and -90 <= lat <= 90:
        return (lon, lat)
    return None


def _extract_objectid(placemark, *, context=None) -> str:
    try:
        for elem in placemark.iter():
            if context is not None:
                context.checkpoint()
            if _matches(elem, "Data") and _attr(elem, "name") == "OBJECTID":
                value_elem = _find_child(elem, "value")
                value = _text(value_elem)
                if value:
                    return value
            if _matches(elem, "SimpleData") and _attr(elem, "name") == "OBJECTID":
                value = _text(elem)
                if value:
                    return value
    except AnalysisCancelled:
        raise
    except Exception:
        return "N/A"
    return "N/A"


def _extract_line_coordinate_paths(placemark, state: _ParserState, *, source: str, feature_name: str):
    paths = []

    for line_elem in _iter_desc(placemark, "LineString", context=state.context):
        if state.context is not None:
            state.context.checkpoint()
        coords_elem = _find_child(line_elem, "coordinates")
        coords = _parse_coordinates_text(
            _text(coords_elem),
            state,
            source=source,
            feature_name=feature_name,
            geometry="LineString",
        )
        if len(coords) >= 2:
            paths.append(coords)
        else:
            _diag(
                state,
                "short_linestring",
                "Skipped a LineString with fewer than two valid coordinates.",
                source=source,
                feature_name=feature_name,
            )

    return paths


def _extract_track_coordinate_paths(placemark, state: _ParserState, *, source: str, feature_name: str):
    paths = []

    for track_elem in _iter_desc(placemark, "Track", GX_NS, context=state.context):
        if state.context is not None:
            state.context.checkpoint()
        coords = []
        invalid_count = 0
        for coord_elem in _iter_desc(track_elem, "coord", GX_NS, context=state.context):
            if state.context is not None:
                state.context.checkpoint()
            try:
                coord = _parse_gx_coord_text(_text(coord_elem))
            except (ValueError, IndexError):
                coord = None
            if coord is None:
                invalid_count += 1
            else:
                coords.append(coord)

        if invalid_count:
            _diag(
                state,
                "invalid_gx_coord",
                f"Rejected gx:Track containing {invalid_count} invalid gx:coord value(s); no connections were inferred across missing vertices.",
                level="error",
                source=source,
                feature_name=feature_name,
                geometry="gx:Track",
            )

            coords = []

        if len(coords) >= 2:
            paths.append(coords)
        else:
            _diag(
                state,
                "short_gx_track",
                "Skipped a gx:Track with fewer than two valid coordinates.",
                source=source,
                feature_name=feature_name,
            )

    return paths


def _extract_point_coords(placemark, state: _ParserState, *, source: str, feature_name: str):
    """Collect actual Point geometries, never the vertices of other shapes.

    The second return value preserves the old first-Point acceptance rule only
    for generated feature names. Expanding pin records must not rename later
    unnamed pipelines, including when a formerly accepted Point is malformed.
    """
    points = []
    first_point_legacy_accepted = False
    for point_index, point_elem in enumerate(_iter_desc(placemark, "Point", context=state.context)):
        coords_text = _text(_find_child(point_elem, "coordinates"))
        coords = _parse_coordinates_text(
            coords_text,
            state,
            source=source,
            feature_name=feature_name,
            geometry="Point",
        )
        if point_index == 0:
            first_point_legacy_accepted = bool(coords)
        tokens = coords_text.split()
        if len(tokens) != 1:
            _diag(
                state,
                "missing_point_coordinate" if not tokens else "ambiguous_point_coordinate",
                "Skipped a Point without a coordinate tuple." if not tokens else
                "Skipped a Point containing multiple coordinate tuples; a pin must have exactly one location.",
                source=source,
                feature_name=feature_name,
                geometry="Point",
            )
            continue
        if not coords:
            continue
        parts = tokens[0].split(",")
        try:
            valid_tuple = len(parts) in (2, 3) and (len(parts) == 2 or math.isfinite(float(parts[2])))
        except ValueError:
            valid_tuple = False
        if not valid_tuple:
            _diag(
                state,
                "invalid_coordinate",
                "Skipped a Point with an invalid coordinate tuple or altitude.",
                level="error",
                source=source,
                feature_name=feature_name,
                geometry="Point",
            )
            continue
        points.append(coords[0])
    return points, first_point_legacy_accepted


def _unsupported_geometry_names(placemark, *, context=None) -> list[str]:
    unsupported = set()
    for elem in placemark.iter():
        if context is not None:
            context.checkpoint()
        local = _local_name(elem)
        if local in {"Polygon", "LinearRing", "Model"}:
            unsupported.add(local)
    return sorted(unsupported)


def _extract_network_links(root, state: _ParserState, *, source: str):
    links = []

    for link_elem in _iter_desc(root, "NetworkLink", context=state.context):
        if state.context is not None:
            state.context.checkpoint()
        name = _text(_find_child(link_elem, "name")) or "NetworkLink"
        href = ""
        for child in list(link_elem):
            if state.context is not None:
                state.context.checkpoint()
            if _matches(child, "Link") or _matches(child, "Url"):
                href = _text(_find_child(child, "href"))
                break
        if not href:
            _diag(
                state,
                "network_link_missing_href",
                "Skipped a NetworkLink without an href.",
                source=source,
                feature_name=name,
            )
            continue
        links.append({"name": name, "href": href, "source": source})

    return links


def _parse_kml_bytes(data: bytes, state: _ParserState, *, source: str, required: bool, validate_structure=True):
    from pipeline_calculator.parsers.repair import RepairFailure, safe_xml_root

    try:
        if state.context is not None:
            state.context.check()
        root = safe_xml_root(data, context=state.context)
        if state.context is not None:
            state.context.check()
    except ET.ParseError as e:
        if required:
            raise ValueError(f"Invalid KML data in {source}: {str(e)}") from e
        _diag(state, "linked_kml_parse_error", f"Could not parse linked KML: {str(e)}", source=source)
        return []
    except RepairFailure as e:
        # Match source-session handling for optional documents. Resource limits
        # and policy failures still abort, and the structural guard below stays
        # outside this catch so ambiguous geometry can never be skipped here.
        if required or e.category not in {"source", "unsupported"}:
            raise
        _diag(state, "linked_kml_parse_error", f"Could not parse linked KML: {str(e)}", source=source)
        return []

    if validate_structure:
        from pipeline_calculator.parsers.repair import validate_geometry_structure
        validate_geometry_structure(root, source=source, context=state.context)

    state.parsed_kml_files.append(source)
    excluded_features = {}

    for placemark in _iter_desc(root, "Placemark", context=state.context):
        if state.context is not None:
            state.context.checkpoint()
        try:
            name = _text(_find_child(placemark, "name"))
            item_index = state.legacy_feature_count + 1
            if not name:
                name = f"Item_{item_index}"

            objectid = _extract_objectid(placemark, context=state.context)
            line_paths = _extract_line_coordinate_paths(placemark, state, source=source, feature_name=name)
            track_paths = _extract_track_coordinate_paths(placemark, state, source=source, feature_name=name)
            coordinate_paths = line_paths + track_paths
            unsupported = _unsupported_geometry_names(placemark, context=state.context)
            point_coords, first_point_legacy_accepted = _extract_point_coords(
                placemark, state, source=source, feature_name=name,
            )

            # Historical naming counted one pipeline, or the first accepted
            # Point in a feature without a pipeline. Keep that namespace stable.
            if coordinate_paths or first_point_legacy_accepted:
                state.legacy_feature_count += 1

            if unsupported:
                code = "ignored_non_centerline_geometry" if coordinate_paths else "unsupported_geometry"
                summary = excluded_features.setdefault(code, {"feature_count": 0, "geometry_types": set()})
                summary["feature_count"] += 1
                summary["geometry_types"].update(unsupported)

            if coordinate_paths:
                state.pipeline_count += 1
                state.pipelines.append(
                    {
                        "id": state.pipeline_count - 1,
                        "placemark_id": (placemark.get("id") or "").strip() or "N/A",
                        "objectid": objectid,
                        "name": name,
                        "coordinates": coordinate_paths[0],
                        "coordinate_paths": coordinate_paths,
                        "source_kml": source,
                    }
                )

            for _ in point_coords:
                if state.context is not None:
                    state.context.checkpoint()
                state.placemark_count += 1
                state.placemarks.append(
                    {
                        "Placemark_ID": objectid if objectid != "N/A" else f"PM_{state.placemark_count}",
                        "Name": name,
                        "Count": 1,
                    }
                )

            if not (coordinate_paths or point_coords or unsupported):
                _diag(
                    state,
                    "no_supported_geometry",
                    "Skipped a Placemark without supported pipeline or point geometry.",
                    source=source,
                    feature_name=name,
                )
        except AnalysisCancelled:
            raise
        except Exception as e:
            _diag(
                state,
                "malformed_placemark",
                f"Skipped malformed Placemark: {str(e)}",
                source=source,
            )

    for code, summary in excluded_features.items():
        _diag(
            state,
            code,
            f"Excluded non-centerline geometry in {summary['feature_count']} Placemark(s) from pipeline mileage.",
            level="info",
            source=source,
            feature_count=summary["feature_count"],
            geometry_types=sorted(summary["geometry_types"]),
        )

    return _extract_network_links(root, state, source=source)


def _href_without_fragment_or_query(href: str) -> str:
    parsed = urlparse((href or "").strip())
    if parsed.scheme and parsed.scheme.lower() not in ("file",):
        return href.strip()
    raw_path = parsed.path if parsed.scheme else href.split("#", 1)[0].split("?", 1)[0]
    return unquote(raw_path.strip())


def _is_remote_href(href: str) -> bool:
    parsed = urlparse((href or "").strip())
    return parsed.scheme.lower() in {"http", "https"}


def _normalize_archive_name(name: str) -> str:
    parts = []
    for part in str(name).replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return ""
        parts.append(part)
    return "/".join(parts)


def _resolve_archive_href(source: str, href: str) -> str:
    raw = urlparse((href or "").strip())
    if raw.scheme or raw.netloc:
        return ""
    href_path = _href_without_fragment_or_query(href).replace("\\", "/")
    if not href_path or href_path.startswith("/"):
        return ""
    # Normalize after joining to the referring document, allowing parent steps
    # inside the archive but rejecting attempts to leave its root. Never extract
    # archive entries or consult external files while resolving these links.
    parts = list(PurePosixPath(source).parent.parts)
    for part in href_path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return ""
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _select_primary_kml(infos: list[zipfile.ZipInfo]) -> zipfile.ZipInfo:
    kml_infos = [info for info in infos if not info.is_dir() and info.filename.lower().endswith(".kml")]
    if not kml_infos:
        raise ValueError("No KML file found in KMZ archive")

    for info in kml_infos:
        if _normalize_archive_name(info.filename).lower() == "doc.kml":
            return info

    nested_doc = [info for info in kml_infos if _normalize_archive_name(info.filename).lower().endswith("/doc.kml")]
    if nested_doc:
        return sorted(nested_doc, key=lambda info: _normalize_archive_name(info.filename).count("/"))[0]

    return max(kml_infos, key=lambda info: int(info.file_size))


def _parse_kmz(path: str, state: _ParserState) -> None:
    with zipfile.ZipFile(path, "r") as kmz:
        infos = kmz.infolist()
        if len(infos) > MAX_ARCHIVE_ENTRIES:
            raise ValueError("KMZ archive exceeds the entry-count limit")
        primary = _select_primary_kml(infos)
        entry_names = {}
        for info in infos:
            if state.context is not None:
                state.context.checkpoint()
            if info.is_dir() or not info.filename.lower().endswith(".kml"):
                continue
            name = _normalize_archive_name(info.filename)
            if not name or info.filename.startswith(("/", "\\")):
                raise ValueError("KMZ contains an invalid KML entry path")
            if name in entry_names:
                raise ValueError(f"KMZ contains ambiguous duplicate KML entries: {name}")
            entry_names[name] = info
        primary_name = _normalize_archive_name(primary.filename)
        _diag(
            state,
            "selected_primary_kml",
            "Selected primary KML from KMZ archive.",
            level="info",
            source=primary_name,
        )

        parsed: set[str] = set()
        queue = deque([primary_name])
        queued = {primary_name}

        while queue:
            if state.context is not None:
                state.context.checkpoint()
            source = queue.popleft()
            if source in parsed:
                continue
            info = entry_names.get(source)
            if info is None:
                _diag(state, "unresolved_network_link", "NetworkLink target was not found in the KMZ.", source=source)
                continue

            parsed.add(source)
            with kmz.open(info) as stream:
                data = _read_document(stream, info.file_size, state)
            links = _parse_kml_bytes(data, state, source=source, required=(source == primary_name))

            for link in links:
                if state.context is not None:
                    state.context.checkpoint()
                href = link["href"]
                if _is_remote_href(href):
                    _diag(
                        state,
                        "remote_network_link_skipped",
                        "Skipped remote NetworkLink; packaged app does not fetch network resources.",
                        source=link["source"],
                        feature_name=link["name"],
                        href=href,
                    )
                    continue

                target = _resolve_archive_href(source, href)
                if not target.lower().endswith(".kml"):
                    _diag(
                        state,
                        "unsupported_network_link_target",
                        "Skipped NetworkLink target that is not a local KML file.",
                        source=link["source"],
                        feature_name=link["name"],
                        href=href,
                        target=target,
                    )
                    continue
                if target not in entry_names:
                    _diag(
                        state,
                        "unresolved_network_link",
                        "NetworkLink target was not found in the KMZ.",
                        source=link["source"],
                        feature_name=link["name"],
                        href=href,
                        target=target,
                    )
                    continue
                if target not in queued:
                    queued.add(target)
                    queue.append(target)

        for name in sorted(entry_names):
            if state.context is not None:
                state.context.checkpoint()
            if name not in parsed:
                _diag(
                    state,
                    "unparsed_kml_file",
                    "KML file exists in the KMZ but was not reachable from the selected primary document.",
                    source=name,
                )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_kml_file(path: str, state: _ParserState) -> None:
    root_path = Path(path).resolve()
    parsed: set[Path] = set()
    queue = deque([root_path])
    queued = {root_path}
    base_dir = root_path.parent

    while queue:
        if state.context is not None:
            state.context.checkpoint()
        current = queue.popleft().resolve()
        if current in parsed:
            continue
        if not _is_within(current, base_dir):
            _diag(
                state,
                "network_link_outside_base_skipped",
                "Skipped local NetworkLink outside the source KML directory.",
                source=str(current),
            )
            continue
        if not current.exists():
            _diag(state, "unresolved_network_link", "NetworkLink target was not found.", source=str(current))
            continue

        parsed.add(current)
        with current.open("rb") as stream:
            data = _read_document(stream, current.stat().st_size, state)
        links = _parse_kml_bytes(data, state, source=str(current), required=(current == root_path))

        for link in links:
            if state.context is not None:
                state.context.checkpoint()
            href = link["href"]
            if _is_remote_href(href):
                _diag(
                    state,
                    "remote_network_link_skipped",
                    "Skipped remote NetworkLink; packaged app does not fetch network resources.",
                    source=link["source"],
                    feature_name=link["name"],
                    href=href,
                )
                continue

            href_path = _href_without_fragment_or_query(href)
            target = (current.parent / href_path).resolve()
            if target.suffix.lower() != ".kml":
                _diag(
                    state,
                    "unsupported_network_link_target",
                    "Skipped NetworkLink target that is not a local KML file.",
                    source=link["source"],
                    feature_name=link["name"],
                    href=href,
                    target=str(target),
                )
                continue
            if target not in queued:
                queued.add(target)
                queue.append(target)


def extract_features_from_file_with_diagnostics(file_path, progress_callback=None, *, context=None) -> ParseResult:
    """Extract pipelines, one row per valid Point pin, and KMZ/KML diagnostics."""
    state = _ParserState(context=context)
    if context is not None:
        context.report("Reading documents")

    try:
        if str(file_path).lower().endswith(".kmz"):
            _parse_kmz(str(file_path), state)
        else:
            _parse_kml_file(str(file_path), state)
    except AnalysisCancelled:
        raise
    except Exception as e:
        raise ValueError(f"Error parsing KML data: {str(e)}") from e

    if not state.pipelines and not state.placemarks:
        _diag(
            state,
            "no_supported_features",
            "No supported pipeline LineString/gx:Track or point Placemark features were found.",
        )

    return ParseResult(
        pipelines=state.pipelines,
        placemarks=state.placemarks,
        diagnostics=state.diagnostics,
        parsed_kml_files=state.parsed_kml_files,
    )


def extract_features_from_file(file_path, progress_callback=None, *, context=None):
    """Extract pipelines and point placemarks from a KMZ/KML file.

    Returns:
      pipelines: list[dict] with keys {id, placemark_id, objectid, name, coordinates, coordinate_paths}
        id is the internal analysis index; placemark_id is the source Placemark's
        id attribute, or N/A when absent. Other identifiers are not substituted.
      placemarks: one row per valid Point geometry with keys {Placemark_ID, Name, Count}.
        Multiple pins may share their source Placemark's name and OBJECTID; every
        Count is 1. Vertices in lines, polygons and rings are never point records.
    """
    result = extract_features_from_file_with_diagnostics(file_path, progress_callback=progress_callback, context=context)
    return result.pipelines, result.placemarks


def parse_kml_kmz_with_diagnostics(file_path, progress_callback=None, *, context=None) -> ParseResult:
    return extract_features_from_file_with_diagnostics(file_path, progress_callback=progress_callback, context=context)


def parse_kml_kmz(file_path, progress_callback=None, *, context=None):
    """Alias for extract_features_from_file (preferred name for the refactor)."""
    return extract_features_from_file(file_path, progress_callback=progress_callback, context=context)
