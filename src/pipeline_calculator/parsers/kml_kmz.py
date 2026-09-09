from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import zipfile
import xml.etree.ElementTree as ET
from urllib.parse import unquote, urlparse


KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"


@dataclass
class ParseResult:
    pipelines: list[dict] = field(default_factory=list)
    placemarks: list[dict] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    parsed_kml_files: list[str] = field(default_factory=list)


@dataclass
class _ParserState:
    pipelines: list[dict] = field(default_factory=list)
    placemarks: list[dict] = field(default_factory=list)
    diagnostics: list[dict] = field(default_factory=list)
    parsed_kml_files: list[str] = field(default_factory=list)
    pipeline_count: int = 0
    placemark_count: int = 0


def _diag(state: _ParserState, code: str, message: str, *, level: str = "warning", **context) -> None:
    entry = {"level": level, "code": code, "message": message}
    clean_context = {k: v for k, v in context.items() if v not in (None, "", [])}
    if clean_context:
        entry["context"] = clean_context
    state.diagnostics.append(entry)


def _tag_uri(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") and "}" in tag else ""


def _local_name(tag_or_elem) -> str:
    tag = getattr(tag_or_elem, "tag", tag_or_elem)
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _matches(elem, local_name: str, namespace: str | None = None) -> bool:
    if _local_name(elem) != local_name:
        return False
    return namespace is None or _tag_uri(elem.tag) == namespace


def _iter_desc(elem, local_name: str, namespace: str | None = None):
    for child in elem.iter():
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


def _extract_objectid(placemark) -> str:
    try:
        for elem in placemark.iter():
            if _matches(elem, "Data") and _attr(elem, "name") == "OBJECTID":
                value_elem = _find_child(elem, "value")
                value = _text(value_elem)
                if value:
                    return value
            if _matches(elem, "SimpleData") and _attr(elem, "name") == "OBJECTID":
                value = _text(elem)
                if value:
                    return value
    except Exception:
        return "N/A"
    return "N/A"


def _extract_line_coordinate_paths(placemark, state: _ParserState, *, source: str, feature_name: str):
    paths = []

    for line_elem in _iter_desc(placemark, "LineString"):
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

    for track_elem in _iter_desc(placemark, "Track", GX_NS):
        coords = []
        invalid_count = 0
        for coord_elem in _iter_desc(track_elem, "coord", GX_NS):
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
    point_elem = next(_iter_desc(placemark, "Point"), None)
    if point_elem is None:
        return []
    coords_elem = _find_child(point_elem, "coordinates")
    return _parse_coordinates_text(
        _text(coords_elem),
        state,
        source=source,
        feature_name=feature_name,
        geometry="Point",
    )


def _unsupported_geometry_names(placemark) -> list[str]:
    unsupported = set()
    for elem in placemark.iter():
        local = _local_name(elem)
        if local in {"Polygon", "LinearRing", "Model"}:
            unsupported.add(local)
    return sorted(unsupported)


def _extract_network_links(root, state: _ParserState, *, source: str):
    links = []

    for link_elem in _iter_desc(root, "NetworkLink"):
        name = _text(_find_child(link_elem, "name")) or "NetworkLink"
        href = ""
        for child in list(link_elem):
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


def _parse_kml_bytes(data: bytes, state: _ParserState, *, source: str, required: bool):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        if required:
            raise ValueError(f"Invalid KML data in {source}: {str(e)}") from e
        _diag(state, "linked_kml_parse_error", f"Could not parse linked KML: {str(e)}", source=source)
        return []

    state.parsed_kml_files.append(source)

    for placemark in _iter_desc(root, "Placemark"):
        try:
            name = _text(_find_child(placemark, "name"))
            item_index = state.pipeline_count + state.placemark_count + 1
            if not name:
                name = f"Item_{item_index}"

            objectid = _extract_objectid(placemark)
            line_paths = _extract_line_coordinate_paths(placemark, state, source=source, feature_name=name)
            track_paths = _extract_track_coordinate_paths(placemark, state, source=source, feature_name=name)
            coordinate_paths = line_paths + track_paths
            unsupported = _unsupported_geometry_names(placemark)

            if coordinate_paths:
                if unsupported:
                    _diag(
                        state,
                        "ignored_non_centerline_geometry",
                        "Ignored non-centerline geometry in a Placemark that also has pipeline path geometry.",
                        source=source,
                        feature_name=name,
                        geometry_types=unsupported,
                    )
                state.pipeline_count += 1
                state.pipelines.append(
                    {
                        "id": state.pipeline_count - 1,
                        "objectid": objectid,
                        "name": name,
                        "coordinates": coordinate_paths[0],
                        "coordinate_paths": coordinate_paths,
                        "source_kml": source,
                    }
                )
                continue

            point_coords = _extract_point_coords(placemark, state, source=source, feature_name=name)
            if point_coords:
                state.placemark_count += 1
                state.placemarks.append(
                    {
                        "Placemark_ID": objectid if objectid != "N/A" else f"PM_{state.placemark_count}",
                        "Name": name,
                        "Count": 1,
                    }
                )
                continue

            if unsupported:
                _diag(
                    state,
                    "unsupported_geometry",
                    "Skipped non-centerline geometry; it was not counted as pipeline mileage.",
                    source=source,
                    feature_name=name,
                    geometry_types=unsupported,
                )
            else:
                _diag(
                    state,
                    "no_supported_geometry",
                    "Skipped a Placemark without supported pipeline or point geometry.",
                    source=source,
                    feature_name=name,
                )
        except Exception as e:
            _diag(
                state,
                "malformed_placemark",
                f"Skipped malformed Placemark: {str(e)}",
                source=source,
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
        primary = _select_primary_kml(infos)
        entry_names = {
            _normalize_archive_name(info.filename): info
            for info in infos
            if not info.is_dir() and info.filename.lower().endswith(".kml")
        }
        primary_name = _normalize_archive_name(primary.filename)
        _diag(
            state,
            "selected_primary_kml",
            "Selected primary KML from KMZ archive.",
            level="info",
            source=primary_name,
        )

        parsed: set[str] = set()
        queue = [primary_name]

        while queue:
            source = queue.pop(0)
            if source in parsed:
                continue
            info = entry_names.get(source)
            if info is None:
                _diag(state, "unresolved_network_link", "NetworkLink target was not found in the KMZ.", source=source)
                continue

            parsed.add(source)
            links = _parse_kml_bytes(kmz.read(info), state, source=source, required=(source == primary_name))

            for link in links:
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
                queue.append(target)

        for name in sorted(entry_names):
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
    queue = [root_path]
    base_dir = root_path.parent

    while queue:
        current = queue.pop(0).resolve()
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
        links = _parse_kml_bytes(current.read_bytes(), state, source=str(current), required=(current == root_path))

        for link in links:
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
            queue.append(target)


def extract_features_from_file_with_diagnostics(file_path, progress_callback=None) -> ParseResult:
    """Extract pipelines, point placemarks, and parser diagnostics from KMZ/KML."""
    state = _ParserState()

    try:
        if str(file_path).lower().endswith(".kmz"):
            _parse_kmz(str(file_path), state)
        else:
            _parse_kml_file(str(file_path), state)
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


def extract_features_from_file(file_path, progress_callback=None):
    """Extract pipelines and point placemarks from a KMZ/KML file.

    Returns:
      pipelines: list[dict] with keys {id, objectid, name, coordinates, coordinate_paths}
      placemarks: list[dict] with keys {Placemark_ID, Name, Count}
    """
    result = extract_features_from_file_with_diagnostics(file_path, progress_callback=progress_callback)
    return result.pipelines, result.placemarks


def parse_kml_kmz_with_diagnostics(file_path, progress_callback=None) -> ParseResult:
    return extract_features_from_file_with_diagnostics(file_path, progress_callback=progress_callback)


def parse_kml_kmz(file_path, progress_callback=None):
    """Alias for extract_features_from_file (preferred name for the refactor)."""
    return extract_features_from_file(file_path, progress_callback=progress_callback)
