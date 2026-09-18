"""Immutable, bounded input sessions for explicitly approved KML repairs.

The original pathname is an identity, never a retry input.  All subsequent
analysis and copy operations consume the captured bytes.  No source archive is
extracted and no scratch pathname is exposed as a feature identity.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import tempfile
import threading
import unicodedata
from urllib.parse import urlparse
import zipfile

from pipeline_calculator.core.execution import AnalysisCancelled
from pipeline_calculator.versioning import get_version
from . import kml_kmz as parser
from .repair import RepairFailure, inspect_document, inspect_geometry, safe_xml_root, validate_repair_encoding, validate_geometry_structure


MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_PACKAGE_BYTES = 256 * 1024 * 1024
MAX_PATCHES = 10_000
MAX_PATCH_BYTES = 1024 * 1024
MAX_DIAGNOSTICS = 10_000
MAX_SOURCE_ELEMENTS = 500_000
CHUNK_BYTES = 1024 * 1024
FORMAT_RULE = "filename_format_mismatch_v1"
_COVERAGE_CODES = parser.INCOMPLETE_CODES | {
    "invalid_coordinate", "invalid_gx_coord", "no_supported_geometry",
    "missing_point_coordinate", "ambiguous_point_coordinate",
}
_SAFE_EXTRA_IDS = {0x0001, 0x000A, 0x5455, 0x7075, 0x6375}


@lru_cache(maxsize=1)
def _application_version():
    return get_version()


def _check(context):
    if context is not None:
        context.check()


def _progress(context, stage, done=0, total=None):
    if context is not None:
        context.report(stage, done, total)


def _digest(data, context=None):
    digest = hashlib.sha256()
    for offset in range(0, len(data), CHUNK_BYTES):
        _check(context)
        digest.update(data[offset:offset + CHUNK_BYTES])
    return digest.hexdigest()


def _failure(code, message, source="", *, category="source", action=None):
    return RepairFailure(message, category=category, findings=[{
        "code": code, "category": "source" if category in {"source", "coverage", "policy"} else category, "source": source,
        "message": message,
        "action": action or (
            "Re-export a complete, self-contained KML/KMZ from the original GIS data, "
            "preserving all pipeline vertices and separate paths."),
        "inspection_coverage": "Validation stopped at this issue; additional issues may remain.",
    }])


def _bounded_findings(findings):
    if len(findings) <= MAX_DIAGNOSTICS:
        return findings
    return findings[:MAX_DIAGNOSTICS - 1] + [{
        "code": "inspection_findings_truncated", "category": "limit",
        "message": "Additional findings were not collected because the inspection limit was reached.",
        "action": "Contact application support about the inspection limit; do not remove geometry.",
    }]


def _read_bounded(path, limit, context=None):
    """Read and stat one open handle, then ensure its pathname still identifies it."""
    _check(context)
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > limit:
            raise _failure("source_size_limit", "Input exceeds the supported input-size limit.",
                           Path(path).name, category="limit",
                           action="Ask application support about the input-size limit; do not remove geometry.")
        chunks = []
        count = 0
        while True:
            _check(context)
            chunk = stream.read(min(CHUNK_BYTES, limit + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > limit:
                raise _failure("source_size_limit", "Input exceeds the supported input-size limit.",
                               Path(path).name, category="limit")
        after = os.fstat(stream.fileno())
    def signature(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
    if signature(before) != signature(after) or signature(after) != signature(Path(path).stat()):
        raise _failure("source_changed", "The source changed while it was being read.",
                       Path(path).name, category="operation", action="Close the exporter and retry this file.")
    return b"".join(chunks)


def _safe_member_name(name):
    # Case folding and NFC are collision checks only: accepted names are never rewritten.
    normalized = parser._normalize_archive_name(name)
    parts = name.replace("\\", "/").rstrip("/").split("/")
    if (not normalized or name.startswith(("/", "\\"))
            or any(part in ("", ".", "..") or ":" in part
                   or part.endswith((".", " ")) or "\x00" in part for part in parts)):
        raise _failure("invalid_archive_path", "The archive contains an unsafe or ambiguous member path.", name)
    return normalized, unicodedata.normalize("NFC", normalized).casefold()


def _safe_extra(extra, source):
    retained = bytearray()
    offset = 0
    while offset < len(extra):
        if offset + 4 > len(extra):
            raise _failure("invalid_archive_metadata", "The archive has malformed entry metadata.", source)
        kind, size = struct.unpack_from("<HH", extra, offset)
        stop = offset + 4 + size
        if stop > len(extra) or kind not in _SAFE_EXTRA_IDS:
            raise _failure("unsupported_archive_metadata", "This archive uses unsupported entry metadata.",
                           source, category="policy",
                           action="Ask the client for a standard ZIP/KMZ export retaining the complete system.")
        if kind != 0x0001:  # ZIP64 sizes/offsets must be regenerated by the writer.
            retained.extend(extra[offset:stop])
        offset = stop
    return bytes(retained)


@dataclass(frozen=True)
class _Member:
    name: str
    canonical: str
    data: bytes
    date_time: tuple
    compress_type: int
    comment: bytes
    extra: bytes
    create_system: int
    external_attr: int
    internal_attr: int
    directory: bool

    def info(self, *, size=None):
        info = zipfile.ZipInfo(self.name, self.date_time)
        info.compress_type = self.compress_type
        info.comment = self.comment
        info.extra = self.extra
        info.create_system = self.create_system
        info.external_attr = self.external_attr
        info.internal_attr = self.internal_attr
        info.file_size = len(self.data) if size is None else size
        return info


def _archive_snapshot(data, context=None):
    """Validate every member, including unreferenced assets, without extraction."""
    if not data.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        raise _failure("archive_preamble", "The ZIP has an unsupported preamble or mixed file content.")
    eocd = data.rfind(b"PK\x05\x06", max(0, len(data) - 65557))
    if eocd < 0 or eocd + 22 > len(data):
        raise _failure("invalid_archive", "The KMZ archive is incomplete or invalid.")
    comment_size = struct.unpack_from("<H", data, eocd + 20)[0]
    if eocd + 22 + comment_size != len(data):
        raise _failure("archive_trailing_content", "The KMZ has unexplained data after its ZIP directory.")
    if struct.unpack_from("<HH", data, eocd + 4) != (0, 0):
        raise _failure("unsupported_archive", "Multi-disk archives are not supported.")
    members = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > parser.MAX_ARCHIVE_ENTRIES:
                raise _failure("archive_entry_limit", "The KMZ exceeds the supported archive entry limit.",
                               category="limit")
            if infos and min(info.header_offset for info in infos) != 0:
                raise _failure("archive_preamble", "The KMZ has an unsupported ZIP preamble.")
            primary = parser._select_primary_kml(infos)
            aliases = {}
            total = 0
            for index, info in enumerate(infos):
                _progress(context, "Checking archive", index, len(infos))
                if info.orig_filename != info.filename:
                    raise _failure("invalid_archive_path", "The archive member name contains a NUL or changes during ZIP decoding.", info.orig_filename)
                canonical, portable = _safe_member_name(info.filename)
                if portable in aliases:
                    raise _failure("ambiguous_archive_path", "The archive contains colliding member paths.", info.filename)
                aliases[portable] = info.is_dir()
                if info.flag_bits & 1:
                    raise _failure("encrypted_archive", "Encrypted KMZ members are unsupported.", info.filename,
                                   category="policy", action="Request an unencrypted export of the complete system.")
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
                                              zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
                    raise _failure("unsupported_compression", "The KMZ uses unsupported ZIP compression.",
                                   info.filename, category="policy")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise _failure("special_archive_member", "The KMZ contains a link or special-file entry.", info.filename)
                if info.file_size < 0 or total + info.file_size > MAX_PACKAGE_BYTES:
                    raise _failure("package_size_limit", "The decompressed KMZ exceeds the repair package limit.",
                                   category="limit")
                extra = _safe_extra(info.extra, info.filename)
                chunks = []
                actual = 0
                with archive.open(info) as stream:
                    while True:
                        _check(context)
                        chunk = stream.read(min(CHUNK_BYTES, MAX_PACKAGE_BYTES - total - actual + 1))
                        if not chunk:
                            break
                        actual += len(chunk)
                        if actual + total > MAX_PACKAGE_BYTES:
                            raise _failure("package_size_limit", "The decompressed KMZ exceeds the repair package limit.",
                                           category="limit")
                        chunks.append(chunk)
                if actual != info.file_size or (info.is_dir() and actual):
                    raise _failure("invalid_archive_metadata", "The archive member size is inconsistent.", info.filename)
                total += actual
                members.append(_Member(info.filename, canonical, b"".join(chunks), tuple(info.date_time),
                                       info.compress_type, info.comment, extra, info.create_system,
                                       info.external_attr, info.internal_attr, info.is_dir()))
            for portable in aliases:
                parts = portable.split("/")
                if any(aliases.get("/".join(parts[:i])) is False for i in range(1, len(parts))):
                    raise _failure("ambiguous_archive_path", "A member path traverses another file entry.", portable)
            return tuple(members), archive.comment, parser._normalize_archive_name(primary.filename)
    except (zipfile.BadZipFile, RuntimeError, EOFError, NotImplementedError) as exc:
        raise _failure("invalid_archive", f"The KMZ could not be verified: {exc}") from exc
    except ValueError as exc:
        if isinstance(exc, RepairFailure):
            raise
        raise _failure("invalid_archive", str(exc)) from exc


def _portable_identity(source, original_path, effective_format):
    if effective_format == "kmz":
        return source
    return Path(source).relative_to(Path(original_path).parent).as_posix()


def _references(root):
    """References that can make a standalone rename/relocation change its meaning."""
    for element in root.iter():
        if "{http://www.w3.org/XML/1998/namespace}base" in element.attrib:
            yield "xml:base", element.attrib["{http://www.w3.org/XML/1998/namespace}base"]
        local = parser._local_name(element)
        if local in {"href", "styleUrl", "targetHref", "sourceHref", "schemaUrl"}:
            yield local, (element.text or "").strip()
        for key, value in element.attrib.items():
            if parser._local_name(key) in {"href", "schemaUrl"}:
                yield parser._local_name(key), value


def _portability(documents, effective_format, members, context=None):
    names = {member.canonical for member in members}
    element_count = 0
    reason = ""
    for source, document in documents:
        root = safe_xml_root(document.effective, context=context)
        for element in root.iter():
            element_count += 1
            if element_count % 256 == 0:
                _check(context)
            if element_count > MAX_SOURCE_ELEMENTS:
                raise _failure("xml_complexity_limit", "The complete source exceeds the supported XML element-count limit.", category="limit")
        for kind, value in _references(root):
            _check(context)
            if not value or (value.startswith("#") and kind != "xml:base"):
                continue
            parsed = urlparse(value)
            if kind == "xml:base":
                reason = "Saving is unavailable because xml:base makes this document location-dependent."
                continue
            if parsed.scheme.lower() in {"http", "https"}:
                # External visual references are preserved, never fetched or certified.
                continue
            if effective_format == "kml":
                reason = reason or "Saving a standalone KML with local resource or document references is not supported."
                continue
            target = parser._resolve_archive_href(source, value)
            if not target or target not in names:
                reason = reason or "Saving is unavailable because a local resource reference is outside or missing from the KMZ."
    return not reason, reason


def _coverage(state, documents, context=None):
    """Compare the independent semantic inventory with the ordinary extraction."""
    expected_pipelines = []
    expected_points = []
    expected_name_slots = 0
    findings = []
    for source, document in documents:
        _check(context)
        root = safe_xml_root(document.effective, context=context)
        # The ordinary resolver does not implement inherited XML Base. A strict
        # XML parse and a matching projection cannot prove the intended link
        # graph when a base URI changes what those links identify.
        if any("{http://www.w3.org/XML/1998/namespace}base" in element.attrib for element in root.iter()):
            findings.append({
                "code": "unsupported_xml_base", "category": "source", "source": source,
                "message": "This document uses xml:base, whose inherited link resolution is not supported by verified repair.",
                "action": "Ask the client to re-export a complete self-contained KML/KMZ with explicitly resolved local links. Preserve the intended linked documents and pipeline geometry; do not simply remove xml:base without correcting its dependent links.",
            })
        inventory = inspect_geometry(root, source=source, context=context)
        findings.extend(inventory["findings"])
        previous_ordinal = None
        for feature in inventory["features"]:
            _check(context)
            kind = feature["kind"]
            if feature["feature_ordinal"] != previous_ordinal:
                name = feature["name"] or f"Item_{expected_name_slots + 1}"
                expected_name_slots += int(feature["legacy_name_slot"])
                previous_ordinal = feature["feature_ordinal"]
            if kind == "pipeline":
                paths = feature["coordinate_paths"]
                expected_pipelines.append({
                    "id": len(expected_pipelines), "placemark_id": feature["placemark_id"],
                    "objectid": feature["objectid"], "name": name,
                    "coordinates": paths[0], "coordinate_paths": paths, "source_kml": source,
                })
            elif kind == "point":
                objectid = feature["objectid"]
                expected_points.append({"Name": name, "Count": 1,
                                        "Placemark_ID": objectid if objectid != "N/A" else f"PM_{len(expected_points) + 1}"})
    # The independent inventory identifies coordinate/path causes precisely. The
    # application emits secondary "short" warnings after rejecting coordinates;
    # do not repeat those as claims that the original path had too few vertices.
    inventory_codes = {"invalid_coordinate", "invalid_gx_coord", "short_linestring", "short_gx_track",
                       "no_supported_geometry", "missing_point_coordinate", "ambiguous_point_coordinate"}
    for item in state.diagnostics:
        if item["code"] in _COVERAGE_CODES - inventory_codes:
            findings.append({"code": item["code"], "category": "source",
                             "source": item.get("context", {}).get("source", ""),
                             "message": item["message"], "context": item.get("context", {}),
                             "action": "Correct the affected source records or include the missing linked files in a complete export."})
    if not state.pipelines:
        findings.append({"code": "no_supported_centerlines", "category": "source",
                         "message": "No supported pipeline LineString or gx:Track paths are available.",
                         "action": "Export the intended pipeline centerlines as KML LineStrings or gx:Tracks; do not substitute polygon outlines."})
    def canonical(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if findings:
        raise RepairFailure("Formatting was repaired, but required geometry or linked files could not be verified.",
                            category="coverage", findings=_bounded_findings(findings))
    if canonical(expected_pipelines) != canonical(state.pipelines) or canonical(expected_points) != canonical(state.placemarks):
        raise _failure("projection_mismatch", "The independent geometry inventory did not match application parsing.",
                       category="verification", action="Provide this diagnostic report to application support.")
    return {"status": "complete", "pipeline_count": len(state.pipelines),
            "point_count": len(state.placemarks), "path_count": sum(len(p["coordinate_paths"]) for p in state.pipelines),
            "vertex_count": sum(len(path) for p in state.pipelines for path in p["coordinate_paths"])}


class SourceSession:
    """Session-owned immutable bytes and baseline; approval is the sole state transition."""
    def __init__(self, *, original_path, original_bytes, effective_format, documents,
                 members, archive_comment, primary, state, report, save_reason=""):
        self.original_path = str(original_path)
        self.display_name = Path(original_path).name
        self.effective_format = effective_format
        self.token = report["source_manifest_sha256"]
        self._original_bytes = original_bytes
        self._documents = tuple(documents)
        self._members = tuple(members)
        self._archive_comment = archive_comment
        self._primary = primary
        self._baseline = json.dumps({"pipelines": state.pipelines, "placemarks": state.placemarks,
                                     "diagnostics": state.diagnostics, "parsed_kml_files": state.parsed_kml_files},
                                    ensure_ascii=False, allow_nan=False)
        self._report = json.dumps(report, ensure_ascii=False, allow_nan=False)
        self._requires_repair = bool(report["rules"])
        self._approved = not self._requires_repair
        self._save_reason = save_reason
        self._lock = threading.RLock()

    @property
    def requires_repair(self):
        return self._requires_repair

    @property
    def approved(self):
        with self._lock:
            return self._approved

    @property
    def verified(self):
        return self.requires_repair and self.approved

    @property
    def report(self):
        result = json.loads(self._report)
        result["status"] = "verified" if self.requires_repair and self.approved else (
            "eligible" if self.requires_repair else "not_needed")
        return result

    @property
    def can_save(self):
        return self.requires_repair and self.approved and not self._save_reason

    @property
    def save_unavailable_reason(self):
        if self._save_reason:
            return self._save_reason
        if not self.requires_repair:
            return "This input did not need repair."
        if not self.approved:
            return "Approve Repair & analyze before saving a verified copy."
        return ""

    @contextmanager
    def lease(self):
        # In-memory immutable ownership is naturally retained by worker references.
        # No file/directory cleanup can invalidate another worker's live lease.
        yield self

    def close(self):
        """Retire the owner's reference without invalidating a worker's snapshot."""
        # The GUI drops its reference. There are no persistent handles or temporary
        # directories, and a running worker owns its own reference to this object.

    def approve(self, *, context=None):
        _progress(context, "Verifying geometry")
        _check(context)
        with self._lock:
            self._approved = True
        return self

    def fresh_parse(self, *, context=None):
        _check(context)
        if not self.approved:
            raise _failure("repair_approval_required", "Repair & analyze must be approved before analysis.",
                           category="operation")
        value = json.loads(self._baseline)
        for pipeline in value["pipelines"]:
            pipeline["coordinate_paths"] = [[tuple(point) for point in path] for path in pipeline["coordinate_paths"]]
            pipeline["coordinates"] = pipeline["coordinate_paths"][0]
        return parser.ParseResult(**value)

    def _candidate_bytes(self, context=None):
        effective = dict((source, document.effective) for source, document in self._documents)
        if self.effective_format == "kml":
            return effective[self._primary]
        if not any(document.edits for _, document in self._documents):
            return self._original_bytes  # Pure routing correction is byte-identical.
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.comment = self._archive_comment
            for member in self._members:
                _check(context)
                data = effective.get(member.canonical, member.data)
                with archive.open(member.info(size=len(data)), "w") as stream:
                    for offset in range(0, len(data), CHUNK_BYTES):
                        _check(context)
                        stream.write(data[offset:offset + CHUNK_BYTES])
        candidate = output.getvalue()
        if len(candidate) > MAX_SOURCE_BYTES:
            raise _failure("saved_copy_size_limit", "The saved archive exceeds the supported package limit.", category="limit")
        return candidate

    def save(self, path, *, context=None):
        """Stage, independently reimport, then publish atomically without replacement."""
        if not self.can_save:
            raise _failure("save_unavailable", self.save_unavailable_reason, category="operation")
        target = Path(path).absolute()
        if target.suffix.lower() != f".{self.effective_format}":
            raise _failure("save_wrong_suffix", f"Save this verified input with the .{self.effective_format} suffix.",
                           category="operation")
        source_paths = [Path(self.original_path)] if self.effective_format == "kmz" else [Path(s) for s, _ in self._documents]
        for source in source_paths:
            if target.resolve() == source.resolve() or (target.exists() and source.exists() and os.path.samefile(target, source)):
                raise _failure("save_source_alias", "A repaired copy cannot replace or alias an original source file.", category="operation")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"The destination already exists: {target.name}")
        _progress(context, "Saving repaired copy")
        candidate = self._candidate_bytes(context)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent)
        staged = Path(temporary)
        published = False
        try:
            with os.fdopen(fd, "wb") as stream:
                for offset in range(0, len(candidate), CHUNK_BYTES):
                    _check(context)
                    stream.write(candidate[offset:offset + CHUNK_BYTES])
                stream.flush()
                os.fsync(stream.fileno())
            _check(context)
            # The staged copy receives the real suffix and ordinary, unpinned selection.
            reopened = prepare_source(staged, context=context)
            if reopened.requires_repair:
                raise _failure("saved_copy_not_idempotent", "The saved copy still requires repair.", category="verification")
            actual = reopened.fresh_parse(context=context)
            expected = self.fresh_parse(context=context)
            if self.effective_format == "kml":
                old_root = self._primary
                for item in actual.pipelines:
                    if item["source_kml"] != str(staged.resolve()):
                        raise _failure("saved_copy_identity", "The standalone saved copy resolved an unexpected document.", category="verification")
                    item["source_kml"] = old_root
                actual.parsed_kml_files = [old_root if s == str(staged.resolve()) else s for s in actual.parsed_kml_files]
            if (reopened._primary != self._primary and self.effective_format == "kmz") or (
                    actual.pipelines != expected.pipelines or actual.placemarks != expected.placemarks
                    or actual.parsed_kml_files != expected.parsed_kml_files):
                raise _failure("saved_copy_selection_changed", "The saved copy selects different source documents or geometry.", category="verification")
            if self.effective_format == "kmz":
                effective = {source: doc.effective for source, doc in self._documents}
                if len(reopened._members) != len(self._members):
                    raise _failure("saved_copy_manifest", "The saved archive member inventory changed.", category="verification")
                for original, saved in zip(self._members, reopened._members):
                    if (saved.name != original.name or saved.data != effective.get(original.canonical, original.data)
                            or saved.date_time != original.date_time or saved.comment != original.comment
                            or saved.external_attr != original.external_attr or saved.internal_attr != original.internal_attr
                            or saved.extra != original.extra or saved.create_system != original.create_system
                            or saved.compress_type != original.compress_type):
                        raise _failure("saved_copy_manifest", "The saved archive did not preserve its supported member metadata and content.", category="verification")
                if reopened._archive_comment != self._archive_comment:
                    raise _failure("saved_copy_manifest", "The saved archive comment changed.", category="verification")
            # Explicit public-parser round trip: no repair routing or source overlay.
            ordinary = parser.extract_features_from_file_with_diagnostics(str(staged), context=context)
            if ordinary.pipelines != reopened.fresh_parse().pipelines or ordinary.placemarks != reopened.fresh_parse().placemarks:
                raise _failure("saved_copy_parser_mismatch", "The ordinary parser did not reproduce the saved input.", category="verification")
            _check(context)
            try:
                os.link(staged, target)
            except FileExistsError:
                raise
            except OSError as exc:
                raise _failure("atomic_save_unavailable", "The destination filesystem could not publish a copy without overwriting.",
                               category="operation", action="Choose another local destination and retry saving.") from exc
            published = True
            # Cancellation after publication cannot make an existing completed file disappear.
            return {"status": "saved", "path": str(target), "sha256": hashlib.sha256(candidate).hexdigest(),
                    "source_manifest_sha256": self.token, "ordinary_reimport_verified": True}
        finally:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                if not published:
                    raise


def _prepare_source(path, *, context=None):
    """Freeze one graph and inspect one candidate; callers explicitly approve repair."""
    original_path = Path(path).resolve()
    if original_path.suffix.lower() not in {".kml", ".kmz"}:
        raise _failure("unsupported_file_type", "Choose a KML or KMZ file.", original_path.name)
    _progress(context, "Checking file")
    original_bytes = _read_bounded(original_path, MAX_SOURCE_BYTES, context)
    is_zip = zipfile.is_zipfile(io.BytesIO(original_bytes))
    effective_format = "kmz" if is_zip or original_bytes.startswith(b"PK") else "kml"
    mismatch = original_path.suffix.lower() != f".{effective_format}"
    members, archive_comment = (), b""
    if effective_format == "kmz":
        members, archive_comment, primary = _archive_snapshot(original_bytes, context)
        source_bytes = {m.canonical: m.data for m in members if not m.directory and m.name.lower().endswith(".kml")}
    else:
        primary = str(original_path)
        source_bytes = {primary: original_bytes}
        if mismatch:
            try:
                original_bytes.decode("utf-8-sig", errors="strict")
            except UnicodeError as exc:
                raise _failure("unsupported_repair_encoding", "Format correction requires an unambiguous UTF-8 KML document.", original_path.name,
                               category="policy") from exc
    state = parser._ParserState(context=context)
    if effective_format == "kmz":
        parser._diag(state, "selected_primary_kml", "Selected primary KML from KMZ archive.", level="info", source=primary)
    queue, queued, visited = deque([primary]), {primary}, set()
    documents, graph, deferred = [], [], []
    total_bytes = patch_count = patch_bytes = 0
    rules = [FORMAT_RULE] if mismatch else []
    while queue:
        _check(context)
        source = queue.popleft()
        if source in visited:
            continue
        visited.add(source)
        if len(visited) > parser.MAX_KML_DOCUMENTS:
            raise _failure("document_count_limit", "The linked source exceeds the document-count limit.", category="limit")
        if effective_format == "kml" and source not in source_bytes:
            candidate_path = Path(source)
            if not parser._is_within(candidate_path, original_path.parent):
                parser._diag(state, "network_link_outside_base_skipped", "Skipped local NetworkLink outside the source KML directory.", source=source)
                continue
            if not candidate_path.exists():
                parser._diag(state, "unresolved_network_link", "NetworkLink target was not found.", source=source)
                continue
            source_bytes[source] = _read_bounded(candidate_path, parser.MAX_KML_BYTES, context)
        data = source_bytes[source]
        total_bytes += len(data)
        if len(data) > parser.MAX_KML_BYTES or total_bytes > parser.MAX_TOTAL_KML_BYTES:
            raise _failure("kml_size_limit", "The linked KML input exceeds the supported size limit.", source, category="limit")
        _progress(context, "Checking documents", len(documents))
        try:
            document = inspect_document(data, source=source, context=context)
        except RepairFailure as exc:
            if source == primary or exc.category not in {"source", "unsupported"}:
                raise
            deferred.append(exc)
            parser._diag(state, "linked_kml_parse_error", str(exc), source=source)
            continue
        documents.append((source, document))
        for rule in document.rules:
            if rule not in rules:
                rules.append(rule)
        patch_count += len(document.edits)
        patch_bytes += sum(len(edit.removed) + len(edit.inserted) for edit in document.edits)
        if patch_count > MAX_PATCHES or patch_bytes > MAX_PATCH_BYTES:
            raise _failure("repair_patch_limit", "The source exceeds the supported repair edit limit.", category="limit")
        # Repaired sources receive the stronger independent coverage check below.
        # Ordinary snapshots must still pass the shared structural guard before
        # any baseline is published, even though their XML already parses.
        links = parser._parse_kml_bytes(document.effective, state, source=source,
                                        required=source == primary, validate_structure=False)
        graph.append(source)
        for link in links:
            _check(context)
            href = link["href"]
            if parser._is_remote_href(href):
                parser._diag(state, "remote_network_link_skipped", "Skipped remote NetworkLink; packaged app does not fetch network resources.",
                             source=source, feature_name=link["name"], href=href)
                continue
            if effective_format == "kmz":
                target = parser._resolve_archive_href(source, href)
            else:
                target = str((Path(source).parent / parser._href_without_fragment_or_query(href)).resolve())
            if not target.lower().endswith(".kml"):
                parser._diag(state, "unsupported_network_link_target", "Skipped NetworkLink target that is not a local KML file.",
                             source=source, feature_name=link["name"], href=href, target=target)
                continue
            if effective_format == "kmz" and target not in source_bytes:
                parser._diag(state, "unresolved_network_link", "NetworkLink target was not found in the KMZ.",
                             source=source, feature_name=link["name"], href=href, target=target)
                continue
            if target not in queued:
                queued.add(target)
                queue.append(target)
        if len(state.diagnostics) > MAX_DIAGNOSTICS:
            raise _failure("diagnostic_limit", "The source exceeds the supported diagnostic work limit.", category="limit")
    if effective_format == "kmz":
        for name in sorted(source_bytes):
            if name not in visited:
                parser._diag(state, "unparsed_kml_file", "KML file exists in the KMZ but was not reachable from the selected primary document.", source=name)
    if not state.pipelines and not state.placemarks:
        parser._diag(state, "no_supported_features", "No supported pipeline LineString/gx:Track or point Placemark features were found.")
    if len(state.diagnostics) > MAX_DIAGNOSTICS:
        raise _failure("diagnostic_limit", "The source exceeds the supported diagnostic work limit.", category="limit")
    effective_kml_bytes = sum(len(document.effective) for _, document in documents)
    if effective_kml_bytes > parser.MAX_TOTAL_KML_BYTES:
        raise _failure("kml_size_limit", "The effective linked KML exceeds the supported size limit.", category="limit")
    if effective_format == "kmz":
        effective_by_source = {name: document.effective for name, document in documents}
        effective_package_bytes = sum(len(effective_by_source.get(member.canonical, member.data)) for member in members)
        if effective_package_bytes > MAX_PACKAGE_BYTES:
            raise _failure("package_size_limit", "The effective KMZ exceeds the supported repair package limit.", category="limit")
    if rules and deferred:
        raise RepairFailure("A required linked document could not be repaired safely.", category="coverage",
                            findings=_bounded_findings([finding for error in deferred for finding in error.findings]))
    can_relocate, save_reason = _portability(documents, effective_format, members, context)
    coverage = {"status": "ordinary_parser", "pipeline_count": len(state.pipelines), "point_count": len(state.placemarks)}
    if rules:
        _progress(context, "Verifying geometry")
        for source, document in documents:
            validate_repair_encoding(document.effective, source=source, context=context)
        coverage = _coverage(state, documents, context)
    else:
        for source, document in documents:
            validate_geometry_structure(safe_xml_root(document.effective, context=context),
                                        source=source, context=context)
    if mismatch and effective_format == "kml" and not can_relocate:
        raise _failure("format_mismatch_dependencies", "A misnamed plaintext KML depends on its original filename or location.",
                       original_path.name, action="Request a correctly named self-contained KML/KMZ export preserving every pipeline.")
    if effective_format == "kmz" and rules:
        effective_sizes = {source: len(document.effective) for source, document in documents}
        candidate_infos = [member.info(size=effective_sizes.get(member.canonical, len(member.data))) for member in members]
        if parser._normalize_archive_name(parser._select_primary_kml(candidate_infos).filename) != primary:
            save_reason = "Saving is unavailable because the repaired archive would select a different primary KML. Session analysis retains the original selection."
    # Independent acquisition validation before approval binds every local dependency.
    if effective_format == "kml":
        for source, captured in source_bytes.items():
            if _read_bounded(source, parser.MAX_KML_BYTES, context) != captured:
                raise _failure("source_changed", "A linked source changed during inspection.", Path(source).name,
                               category="operation", action="Close the exporter and retry the complete input.")
    elif _read_bounded(original_path, MAX_SOURCE_BYTES, context) != original_bytes:
        raise _failure("source_changed", "The KMZ changed during inspection.", original_path.name,
                       category="operation", action="Close the exporter and retry the file.")
    manifest = [{"source": _portable_identity(source, original_path, effective_format),
                 "original_sha256": _digest(document.original, context),
                 "effective_sha256": _digest(document.effective, context)} for source, document in documents]
    source_hash = _digest(original_bytes, context)
    identity = {"original_sha256": source_hash, "primary": _portable_identity(primary, original_path, effective_format),
                "documents": manifest}
    report_documents = []
    for source, document in documents:
        entry = document.to_report()
        entry["source"] = _portable_identity(source, original_path, effective_format)
        report_documents.append(entry)
    archive_manifest = [{"source": member.name, "sha256": _digest(member.data, context), "bytes": len(member.data),
                         "xml_inspected": member.canonical in graph} for member in members]
    report = {"schema_version": 1, "version": 1, "verifier_version": 1, "app_version": _application_version(),
              "status": "eligible" if rules else "not_needed", "original_name": original_path.name,
              "source_filename": original_path.name, "source_sha256": source_hash,
              "original_sha256": source_hash, "source_manifest_sha256": _digest(json.dumps(identity, sort_keys=True).encode()),
              "supplied_extension": original_path.suffix.lower(), "effective_format": effective_format,
              "rules": rules, "edit_count": patch_count, "edit_byte_count": patch_bytes,
              "original_preserved": True, "geometry_verification": "verified" if rules else "not_needed",
              "coverage": coverage, "primary": identity["primary"], "documents": report_documents,
              "manifest": manifest, "parsed_documents": [_portable_identity(s, original_path, effective_format) for s in graph],
              "archive_members": archive_manifest,
              "uninspected_documents": [name for name in source_bytes if name not in graph] if effective_format == "kmz" else [],
              "findings": [], "save_unavailable_reason": save_reason,
              "external_visual_assets_verified": False,
              "limits": {"source_bytes": MAX_SOURCE_BYTES, "package_bytes": MAX_PACKAGE_BYTES,
                         "document_bytes": parser.MAX_KML_BYTES, "patches": MAX_PATCHES, "xml_elements": MAX_SOURCE_ELEMENTS}}
    return SourceSession(original_path=original_path, original_bytes=original_bytes, effective_format=effective_format,
                         documents=documents, members=members, archive_comment=archive_comment, primary=primary,
                         state=state, report=report, save_reason=save_reason)


def prepare_source(path, *, context=None):
    """Prepare one source, retaining relative identities in copyable findings."""
    try:
        return _prepare_source(path, context=context)
    except RepairFailure as exc:
        base = Path(path).resolve().parent
        def portable(value):
            if not isinstance(value, str):
                return value
            candidate = Path(value)
            if not candidate.is_absolute():
                return value
            try:
                return candidate.relative_to(base).as_posix()
            except ValueError:
                return candidate.name
        for finding in exc.findings:
            for key in ("source", "target"):
                if key in finding:
                    finding[key] = portable(finding[key])
            if isinstance(finding.get("context"), dict):
                for key in ("source", "target"):
                    if key in finding["context"]:
                        finding["context"][key] = portable(finding["context"][key])
        raise
