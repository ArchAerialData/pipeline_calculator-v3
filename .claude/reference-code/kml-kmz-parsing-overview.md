# KML/KMZ Parsing — Reference Overview

**Source repository:** `pipeline_calculator-v3` — https://github.com/ArchAerialData/pipeline_calculator-v3
**Commit:** `5da6b8677c86b3d6a856509526df68cac5be68b7` (branch `codex/pipeline-ci-signing-setup`)
**Repo root on author's machine:** `C:\Users\rbake\Desktop\GitHub Shortcuts\.vscode repos\pipeline_calculator-v3`

> All file paths below are **repo-relative** (from the repo root above) unless marked absolute. If you're working in a different repo/checkout, these paths will not resolve on disk — use them only as identifiers/citations back to the source repo (e.g. via the GitHub URL + commit hash). Full source of the primary parser module is embedded below so this document is self-contained and does not require access to the original repo.

---

## Summary

All KML/KMZ parsing lives in a single module: `src/pipeline_calculator/parsers/kml_kmz.py`. It uses stdlib `xml.etree.ElementTree` (namespace-agnostic matching, no `lxml` dependency) and `zipfile` for `.kmz` archives. Every other file in the codebase that needs parsed pipeline/placemark data calls into this module — there is no duplicate/alternate parsing implementation.

### Entry points (all wrap the same implementation)
- `extract_features_from_file_with_diagnostics(file_path, progress_callback=None) -> ParseResult` — full result: pipelines, placemarks, diagnostics, parsed_kml_files
- `extract_features_from_file(file_path, progress_callback=None)` — legacy `(pipelines, placemarks)` tuple form
- `parse_kml_kmz` / `parse_kml_kmz_with_diagnostics` — newer-name aliases, re-exported via `src/pipeline_calculator/parsers/__init__.py`

### File handling
- `.kmz` → `_parse_kmz()`: opens as a zip archive, selects the primary KML via `_select_primary_kml()` (prefers a top-level `doc.kml`, then the shallowest nested `doc.kml`, then falls back to the largest `.kml` by file size), then BFS-walks local `NetworkLink` targets found inside the archive.
- `.kml` → `_parse_kml_file()`: same BFS walk pattern but over the filesystem, constrained to stay within the source file's directory (`_is_within()` — a path-traversal guard).
- Zip-entry path traversal (`../`) is normalized/blocked in `_normalize_archive_name()`.

### Geometry extraction (per `Placemark`, inside `_parse_kml_bytes()`)
- **LineString** → `_extract_line_coordinate_paths()` — becomes pipeline centerline geometry.
- **gx:Track** (Google Earth extension namespace) → `_extract_track_coordinate_paths()` — also treated as pipeline geometry.
- **Point** → `_extract_point_coords()` — becomes a "placemark" record (not a pipeline).
- **Polygon / LinearRing / Model** → explicitly unsupported; detected via `_unsupported_geometry_names()` and reported as diagnostics rather than silently dropped.
- A single Placemark can contain multiple LineStrings/Tracks — all are collected into one pipeline record's `coordinate_paths` list (the first path also fills the legacy `coordinates` field for backwards compatibility).

### Coordinate filtering
- `_parse_coordinates_text()` parses `lon,lat[,alt]` tuples, drops out-of-range values (`-180..180` longitude, `-90..90` latitude) and malformed tuples, and counts invalid tuples for diagnostics.
- Any coordinate path with fewer than 2 valid points is dropped entirely (`short_linestring` / `short_gx_track` diagnostic codes).

### NetworkLink resolution
- `_extract_network_links()` finds `NetworkLink` elements and their `href` (via `Link`/`Url` child).
- `_resolve_archive_href()` resolves relative hrefs against the referencing KML's location inside the archive/filesystem.
- Local links are followed recursively (BFS, deduplicated via a `parsed` set to avoid cycles).
- Remote links (`http://`/`https://`) are explicitly **skipped** — the packaged app does not fetch network resources — and a `remote_network_link_skipped` diagnostic is recorded.
- Unresolvable targets, non-`.kml` targets, and out-of-bounds local targets are all diagnosed rather than raising.

### OBJECTID extraction
`_extract_objectid()` pulls the value from ExtendedData `Data`/`value` or `SimpleData` elements named `OBJECTID`; falls back to `"N/A"`.

### Diagnostics system
Every skip/anomaly (invalid coordinates, short lines/tracks, unsupported geometry, unresolved/remote/malformed NetworkLinks, XML parse errors, malformed placemarks, unparsed-but-present KML files in a KMZ) is recorded via `_diag()` into a structured list of `{level, code, message, context}` dicts, returned as `ParseResult.diagnostics` and surfaced in the GUI (diagnostics tab / results page).

---

## Downstream consumers (repo-relative paths, not embedded here)

- `src/pipeline_calculator/core/analyzer.py` — `PipelineAnalyzer.analyze_complete()` calls `extract_features_from_file_with_diagnostics()`, feeds `pipelines` into length calculation (`calculate_pipeline_lengths()`) and overlap/parallel-segment detection; diagnostics pass straight through to the result dict returned to the GUI.
- `src/pipeline_calculator/core/coordinates.py` — `coordinate_paths_for_pipeline()` normalizes a pipeline's `coordinate_paths` (with fallback to the legacy single `coordinates` field) for use by length calculation and segmentation. Full source embedded below (it's short).
- `src/pipeline_calculator/gui/pages/file_select_page.py`, `src/pipeline_calculator/gui/tabs/placemarks_tab.py`, `src/pipeline_calculator/gui/pages/results_page.py` — GUI display of parsed pipelines/placemarks/diagnostics.
- `src/pipeline_calculator/export/corridor_kml.py` — **writes** KML for export; not part of the parsing path.
- `src/pipeline_calculator_v3.py` (line ~59) — legacy compatibility shim, `PipelineCalculator.extract_features_from_file()` just delegates to the same `extract_features_from_file()`.

---

## Full source: `src/pipeline_calculator/parsers/kml_kmz.py`

```python
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
            f"Skipped {invalid_count} invalid coordinate tuple(s).",
            source=source,
            feature_name=feature_name,
            geometry=geometry,
        )

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
                f"Skipped {invalid_count} invalid gx:coord value(s).",
                source=source,
                feature_name=feature_name,
                geometry="gx:Track",
            )

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
    href_path = _normalize_archive_name(_href_without_fragment_or_query(href))
    if not href_path:
        return ""
    source_parent = PurePosixPath(source).parent
    if str(source_parent) == ".":
        resolved = PurePosixPath(href_path)
    else:
        resolved = source_parent / href_path
    return _normalize_archive_name(str(resolved))


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
```

---

## Full source: `src/pipeline_calculator/parsers/__init__.py`

```python
"""Parsers (refactor-in-progress)."""

from pipeline_calculator.parsers.kml_kmz import (
    ParseResult,
    extract_features_from_file,
    extract_features_from_file_with_diagnostics,
    parse_kml_kmz,
    parse_kml_kmz_with_diagnostics,
)

__all__ = [
    "ParseResult",
    "extract_features_from_file",
    "extract_features_from_file_with_diagnostics",
    "parse_kml_kmz",
    "parse_kml_kmz_with_diagnostics",
]
```

---

## Full source: `src/pipeline_calculator/core/coordinates.py`

Normalizes parsed pipeline records (`coordinate_paths` / legacy `coordinates`) for downstream length/segmentation code.

```python
from __future__ import annotations

from pipeline_calculator.core.segmentation import segment_pipeline


def coordinate_paths_for_pipeline(pipeline):
    """Return valid coordinate paths for a pipeline with legacy fallback."""
    paths = []

    for path in pipeline.get("coordinate_paths") or []:
        coords = list(path or [])
        if len(coords) >= 2:
            paths.append(coords)

    if paths:
        return paths

    coords = list(pipeline.get("coordinates") or [])
    return [coords] if len(coords) >= 2 else []


def segment_pipeline_paths(geod, pipeline, segment_length):
    """Segment every coordinate path without connecting disjoint parts."""
    segments = []

    for path_index, coords in enumerate(coordinate_paths_for_pipeline(pipeline)):
        path_segments = segment_pipeline(geod, coords, segment_length)
        for path_segment_index, segment in enumerate(path_segments):
            segment_with_path = dict(segment)
            segment_with_path["path_index"] = path_index
            segment_with_path["path_segment_index"] = segment.get(
                "segment_index",
                path_segment_index,
            )
            segment_with_path["segment_index"] = len(segments)
            segments.append(segment_with_path)

    return segments
```

---

## Relevant excerpt: `src/pipeline_calculator/core/analyzer.py` (lines 1–170 of the file)

Shows how the parser's output feeds into the rest of the analysis pipeline (`PipelineAnalyzer`). This excerpt omits the tail of the file (overlap-result post-processing return statement continuation), which is not part of the parsing path.

```python
from __future__ import annotations

from pipeline_calculator.core.constants import (
    ANGULAR_TOLERANCE,
    DEFAULT_DETECTION_RANGE,
    GEOD_ELLPS,
    MIN_PARALLEL_LENGTH,
    SEGMENT_LENGTH,
    SURVEY_MILE_METERS,
)
from pipeline_calculator.core.coordinates import coordinate_paths_for_pipeline
from pipeline_calculator.core.effective_length import compute_effective_length_by_clusters
from pipeline_calculator.core.overlap import calculate_overlap_results, find_parallel_segments
from pipeline_calculator.core.segmentation import segment_pipeline
from pipeline_calculator.parsers.kml_kmz import (
    extract_features_from_file,
    extract_features_from_file_with_diagnostics,
)


class PipelineAnalyzer:
    """Combined pipeline length and overlap analyzer (package implementation)."""

    def __init__(
        self,
        *,
        geod=None,
        survey_mile=SURVEY_MILE_METERS,
        detection_range=DEFAULT_DETECTION_RANGE,
        min_parallel_length=MIN_PARALLEL_LENGTH,
        segment_length=SEGMENT_LENGTH,
        angular_tolerance=ANGULAR_TOLERANCE,
    ):
        if geod is None:
            from pyproj import Geod

            geod = Geod(ellps=GEOD_ELLPS)

        self.geod = geod
        self.survey_mile = float(survey_mile)
        self.detection_range = float(detection_range)
        self.min_parallel_length = float(min_parallel_length)
        self.segment_length = float(segment_length)
        self.angular_tolerance = float(angular_tolerance)

    def extract_features_from_file(self, file_path, progress_callback=None):
        return extract_features_from_file(file_path, progress_callback=progress_callback)

    def calculate_pipeline_lengths(self, pipelines):
        pipeline_data = []
        total_length_meters = 0.0
        total_length_miles = 0.0

        for pipeline in pipelines:
            length_meters = 0.0
            paths = coordinate_paths_for_pipeline(pipeline)

            if not paths:
                continue

            for coords in paths:
                for i in range(len(coords) - 1):
                    try:
                        lon1, lat1 = coords[i]
                        lon2, lat2 = coords[i + 1]
                        _, _, distance = self.geod.inv(lon1, lat1, lon2, lat2)
                        length_meters += abs(distance)
                    except Exception as e:
                        print(f"Warning: Error calculating distance for pipeline {pipeline.get('name', '')}: {str(e)}")
                        continue

            length_miles = length_meters / self.survey_mile

            pipeline_data.append(
                {
                    "OBJECTID": pipeline.get("objectid", "N/A"),
                    "Name": pipeline.get("name", ""),
                    "Shape_Length": length_meters,
                    "pipelinelength": length_miles,
                }
            )

            total_length_meters += length_meters
            total_length_miles += length_miles

        return pipeline_data, total_length_meters, total_length_miles

    def segment_pipeline(self, coordinates):
        return segment_pipeline(self.geod, coordinates, self.segment_length)

    def find_parallel_segments(self, pipelines, progress_callback=None):
        return find_parallel_segments(
            pipelines,
            geod=self.geod,
            segment_length=self.segment_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
        )

    def calculate_overlap_results(self, pipelines, parallel_groups, progress_callback=None):
        return calculate_overlap_results(
            pipelines,
            parallel_groups,
            geod=self.geod,
            survey_mile_m=self.survey_mile,
            segment_length=self.segment_length,
            min_parallel_length=self.min_parallel_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
        )

    def compute_effective_length_by_clusters(self, pipelines, per_pipeline_total_meters, progress_callback=None):
        return compute_effective_length_by_clusters(
            pipelines,
            per_pipeline_total_meters,
            geod=self.geod,
            segment_length=self.segment_length,
            detection_range=self.detection_range,
            angular_tolerance=self.angular_tolerance,
            progress_callback=progress_callback,
        )

    def analyze_complete(self, file_path, progress_callback=None):
        """Complete analysis of KMZ/KML file."""
        try:
            parsed = extract_features_from_file_with_diagnostics(file_path, progress_callback=progress_callback)
            pipelines = parsed.pipelines
            placemarks = parsed.placemarks

            pipeline_data, total_meters, total_miles = self.calculate_pipeline_lengths(pipelines)

            overlap_results = None
            if len(pipelines) >= 2:
                try:
                    parallel_groups = self.find_parallel_segments(pipelines, progress_callback)
                    overlap_results = self.calculate_overlap_results(pipelines, parallel_groups, progress_callback)
                    per_pipe_totals = [d["Shape_Length"] for d in pipeline_data]
                    eff_total_m = self.compute_effective_length_by_clusters(pipelines, per_pipe_totals, progress_callback)

                    eff_total_m = max(0.0, min(float(total_meters), float(eff_total_m)))
                    total_savings = max(0.0, float(total_meters) - eff_total_m)

                    overlap_results["effective_total_meters"] = eff_total_m
                    overlap_results["effective_total_miles"] = eff_total_m / self.survey_mile
                    overlap_results["savings_meters"] = total_savings
                    overlap_results["savings_miles"] = total_savings / self.survey_mile
                    overlap_results["savings_percentage"] = (
                        (total_savings / total_meters * 100) if total_meters > 0 else 0
                    )
                    overlap_results["computation_method"] = "clustered_segments_v1"
                except Exception as e:
                    print(f"Warning: Overlap analysis failed: {str(e)}")
                    overlap_results = None

            return {
                "pipelines": pipeline_data,
                "placemarks": placemarks,
                "total_meters": total_meters,
                "total_miles": total_miles,
                "overlap_analysis": overlap_results,
                "diagnostics": parsed.diagnostics,
                "parsed_kml_files": parsed.parsed_kml_files,
                "analysis_parameters": {
                    "detection_range": self.detection_range,
                    "min_parallel_length": self.min_parallel_length,
                    "segment_length": self.segment_length,
                    "angular_tolerance": self.angular_tolerance,
                },
                # ... (remainder of return dict / exception handling omitted — not part of the parsing path)
```

---

## Notes for the receiving agent

- This document was generated from a **different repository checkout** than the one you're working in. Don't attempt to resolve the repo-relative paths above against your local filesystem — they're citations back to `pipeline_calculator-v3` at the commit noted at the top.
- The embedded source for `kml_kmz.py`, `parsers/__init__.py`, and `coordinates.py` is complete/verbatim as of that commit. The `analyzer.py` excerpt is partial (lines 1–170) and covers only the parsing-adjacent portion of that file.
- If you need the omitted parts of `analyzer.py`, or the GUI consumer files (`file_select_page.py`, `placemarks_tab.py`, `results_page.py`, `corridor_kml.py`), ask the author to export those separately — they were not included here since they're not part of the core parsing logic.
