"""Conservative XML formatting repairs, with reversible byte-level evidence.

This module never reads files or repairs geometry.  A valid XML document bypasses
repair; coverage of a repaired source is a separate decision made by its owner.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import hashlib
import math
import xml.etree.ElementTree as ET
from xml.parsers import expat

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XML_NS = "http://www.w3.org/XML/1998/namespace"
MAX_XML_BYTES = 64 * 1024 * 1024
MAX_ELEMENTS = 500_000
MAX_DEPTH = 256
MAX_ATTRIBUTES = 256
MAX_PATCHES = 10_000
MAX_PATCH_BYTES = 1024 * 1024
MAX_FINDINGS = 10_000
CHUNK_BYTES = 64 * 1024
_SPACE = b" \t\r\n"
_BOM = b"\xef\xbb\xbf"
_DECLARATION = b' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
_LEAVES = {"name", "description", "Snippet"}
_CONTAINERS = {"Document", "Folder", "Placemark"}
_GEOMETRY = {"Point", "LineString", "LinearRing", "Polygon", "MultiGeometry", "Model"}
_RESERVED = _GEOMETRY | {"kml", "Document", "Folder", "Placemark", "coordinates", "NetworkLink"}
_GX_RESERVED = {"Track", "MultiTrack", "coord"}


def _check(context):
    if context is not None:
        context.check()


def _finding(code, message, source="", *, action=None, category="source", offset=None, data=None):
    item = {"code": code, "message": message, "category": category, "source": source}
    if action:
        item["action"] = action
    if offset is not None and data is not None:
        prefix = data[:offset].replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        item["line"] = prefix.count(b"\n") + 1
        final_line = prefix.rsplit(b"\n", 1)[-1]
        try:
            item["column"] = len(final_line.decode("utf-8")) + 1
        except UnicodeDecodeError:
            item["column"] = len(final_line) + 1
            item["column_unit"] = "bytes"
        item["byte_offset"] = offset
    return item


class RepairFailure(ValueError):
    def __init__(self, message, *, findings=None, category="source"):
        super().__init__(message)
        self.category = category
        self.findings = list(findings or [_finding("repair_unavailable", str(message), category=category)])

    def client_request(self):
        issues = [f for f in self.findings if f.get("category", "source") == "source" or (f.get("category") == "policy" and f.get("action"))]
        if not issues:
            return "This is an application, access, or inspection-limit issue. Contact support; the file has not been shown to contain damaged geometry."
        lines = ["Please ask the client to provide a new complete KML/KMZ that fixes the following issues:", ""]
        for issue in issues[:MAX_FINDINGS]:
            where = issue.get("source") or "Selected file"
            if issue.get("line") is not None:
                where += f", line {issue['line']}"
                if issue.get("column") is not None:
                    where += f", column {issue['column']}"
            action = issue.get("action", "Re-export a complete, valid KML/KMZ from the original GIS data.")
            lines.append(f"- {where}: {issue['message']} {action}")
        lines.extend(["", "Please re-export from the original GIS data while preserving all pipeline vertices, separate paths, and the complete system. Do not delete problematic geometry or guess replacement coordinates. Inspection stopped where the file could not be interpreted safely; additional issues may remain."])
        return "\n".join(lines)


def _fail(code, message, source="", *, category="source", action=None, offset=None, data=None):
    raise RepairFailure(message, category=category, findings=[_finding(code, message, source, category=category, action=action, offset=offset, data=data)])


@dataclass(frozen=True)
class RepairEdit:
    rule: str
    offset: int
    removed: bytes
    inserted: bytes

    def to_report(self):
        return {"rule": self.rule, "byte_offset": self.offset, "removed": self.removed.decode("utf-8"), "inserted": self.inserted.decode("utf-8"), "removed_bytes": len(self.removed), "inserted_bytes": len(self.inserted)}


@dataclass(frozen=True)
class DocumentRepair:
    source: str
    original: bytes
    effective: bytes
    edits: tuple[RepairEdit, ...] = ()
    rules: tuple[str, ...] = ()

    def to_report(self):
        return {"source": self.source, "original_sha256": hashlib.sha256(self.original).hexdigest(), "effective_sha256": hashlib.sha256(self.effective).hexdigest(), "rules": list(self.rules), "edits": [e.to_report() for e in self.edits], "byte_preservation": "passed", "status": "repaired" if self.edits else "not_needed"}


def _parse_error(error):
    result = ET.ParseError(str(error))
    result.code = error.code
    result.position = (error.lineno, error.offset)
    return result


def _preflight(data, *, context=None, start=None, end=None, declaration=None):
    _check(context)
    if len(data) > MAX_XML_BYTES:
        _fail("xml_size_limit", "The document exceeds the 64 MiB XML inspection limit.", category="limit")
    parser = expat.ParserCreate()  # Namespace-free inspection never resolves entities.
    depth = count = 0

    def on_start(name, attrs):
        nonlocal depth, count
        depth += 1
        count += 1
        if depth > MAX_DEPTH or count > MAX_ELEMENTS or len(attrs) > MAX_ATTRIBUTES:
            _fail("xml_complexity_limit", "The document exceeds the supported XML complexity limit.", category="limit")
        if count % 128 == 0:
            _check(context)
        if start:
            start(name, attrs, parser.CurrentByteIndex)

    def on_end(name):
        nonlocal depth
        if end:
            end(name, parser.CurrentByteIndex)
        depth -= 1

    def reject_dtd(*_):
        _fail("unsupported_dtd", "DTD/entity declarations are unsupported; external definitions cannot be verified safely.", action="Export self-contained KML without a DTD or custom entity declarations.")

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.StartDoctypeDeclHandler = reject_dtd
    parser.EntityDeclHandler = reject_dtd
    parser.ExternalEntityRefHandler = reject_dtd
    parser.XmlDeclHandler = declaration
    try:
        for offset in range(0, len(data), CHUNK_BYTES):
            _check(context)
            parser.Parse(data[offset:offset + CHUNK_BYTES], False)
        parser.Parse(b"", True)
        _check(context)
    except expat.ExpatError as error:
        raise _parse_error(error) from error
    except LookupError as error:
        _fail("unsupported_xml_encoding", f"The declared XML encoding is unsupported: {error}", category="policy", action="Re-export the complete source as correctly declared UTF-8 KML/KMZ.")


def safe_xml_root(data: bytes, *, context=None):
    """Apply input-policy bounds before building an ordinary strict XML tree."""
    _preflight(data, context=context)
    _check(context)
    root = ET.fromstring(data)
    _check(context)
    return root


@dataclass
class _Token:
    name: str
    start: int
    start_end: int
    close_start: int
    end: int
    parent: int | None
    plain: bool = True
    empty: bool = False


def _scan(data, *, context=None):
    """Locate tokens only; Expat and ET, never this locator, validate XML grammar."""
    tokens, stack = [], []
    index = 0
    operations = 0
    while index < len(data):
        operations += 1
        if operations % 128 == 0:
            _check(context)
        opening = data.find(b"<", index)
        if opening < 0:
            break
        if data.startswith(b"<!--", opening):
            closing = data.find(b"-->", opening + 4)
            width = 3
        elif data.startswith(b"<![CDATA[", opening):
            closing = data.find(b"]]>", opening + 9)
            width = 3
        elif data.startswith(b"<?", opening):
            closing = data.find(b"?>", opening + 2)
            width = 2
        else:
            closing = -2
            width = 1
        if closing != -2:
            if closing < 0:
                _fail("malformed_xml", "An XML comment, CDATA section, or processing instruction is incomplete.")
            if stack:
                tokens[stack[-1]].plain = False
            index = closing + width
            continue
        if data.startswith(b"<!", opening):
            _fail("unsupported_declaration", "XML declarations other than comments and CDATA cannot be repaired safely.", action="Re-export self-contained KML without DTD declarations or damaged markup.")
        position = opening + 1
        quote = None
        while position < len(data):
            char = data[position]
            if quote:
                if char == quote:
                    quote = None
            elif char in (34, 39):
                quote = char
            elif char == 62:
                break
            if (position - opening) % CHUNK_BYTES == 0:
                _check(context)
            position += 1
        if position >= len(data):
            _fail("malformed_xml", "An XML tag is incomplete.")
        body = data[opening + 1:position]
        is_end = body.startswith(b"/")
        if is_end:
            name = body[1:].strip().decode("utf-8")
            if not stack or tokens[stack[-1]].name != name:
                _fail("malformed_xml", "XML opening and closing tags do not match.", offset=opening, data=data)
            token = tokens[stack.pop()]
            token.close_start, token.end = opening, position + 1
        else:
            name_end = 0
            while name_end < len(body) and body[name_end] not in b" \t\r\n/>":
                name_end += 1
            if not name_end:
                _fail("malformed_xml", "An XML element name is missing.")
            name = body[:name_end].decode("utf-8")
            empty = body.rstrip().endswith(b"/")
            parent = stack[-1] if stack else None
            if parent is not None:
                tokens[parent].plain = False
            tokens.append(_Token(name, opening, position + 1, position, position + 1, parent, empty=empty))
            if len(tokens) > MAX_ELEMENTS or len(stack) >= MAX_DEPTH:
                _fail("xml_complexity_limit", "The document exceeds the supported XML complexity limit.", category="limit")
            if not empty:
                stack.append(len(tokens) - 1)
        index = position + 1
    if stack:
        _fail("malformed_xml", "The XML document is truncated or contains an unclosed element.")
    if not tokens:
        _fail("missing_kml", "No XML root element was found.")
    return tokens


def _local(name):
    return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _split(tag):
    if tag.startswith("{"):
        uri, local = tag[1:].split("}", 1)
        return uri, local
    return "", tag


def _utf8_document(data, source):
    try:
        data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        _fail("unsupported_repair_encoding", "Automatic repair requires unambiguous UTF-8; this document's encoding cannot be verified.", source, action="Re-export the complete source as correctly declared UTF-8 KML/KMZ.")
    if data.startswith(_BOM + _BOM):
        _fail("duplicate_bom", "The document contains more than one leading encoding signature.", source)


def validate_repair_encoding(data, source="", *, context=None):
    """Verify repair's stricter encoding policy, including format-only corrections.

    Unlike ordinary parsing, this never accepts UTF-16 or an encoding declaration
    that happens to parse ASCII bytes despite claiming a different character set.
    """
    _utf8_document(data, source)

    def declaration(version, encoding, standalone):
        if version != "1.0" or (encoding and encoding.lower().replace("-", "") != "utf8"):
            _fail("unsupported_repair_encoding", "Automatic repair requires an XML 1.0 UTF-8 declaration consistent with the bytes.", source)

    _preflight(data, context=context, declaration=declaration)


def _leading_patch(data):
    start = len(_BOM) if data.startswith(_BOM) else 0
    end = start
    while end < len(data) and data[end] in _SPACE:
        end += 1
    if end > start and data.startswith(b"<?xml", end) and data[end + 5:end + 6] in (b" ", b"\t", b"\r", b"\n"):
        return RepairEdit("leading_xml_whitespace_v1", start, data[start:end], b"")
    return None


def _apply(original, edits):
    pieces = []
    position = 0
    for edit in edits:
        if edit.offset < position or edit.offset > len(original) or original[edit.offset:edit.offset + len(edit.removed)] != edit.removed:
            _fail("invalid_patch_ledger", "Repair edits overlap or do not match the original bytes.", category="verification")
        pieces.extend((original[position:edit.offset], edit.inserted))
        position = edit.offset + len(edit.removed)
    pieces.append(original[position:])
    return b"".join(pieces)


def _check_patch_limits(edits, source=""):
    if len(edits) > MAX_PATCHES or sum(len(e.removed) + len(e.inserted) for e in edits) > MAX_PATCH_BYTES:
        _fail("repair_patch_limit", "The proposed repair exceeds this version's edit limits.", source, category="limit")


def _map_to_original(offset, edits):
    delta = 0
    for edit in edits:
        candidate_offset = edit.offset + delta
        if offset < candidate_offset:
            break
        if offset < candidate_offset + len(edit.inserted):
            return edit.offset
        delta += len(edit.inserted) - len(edit.removed)
    return offset - delta


def _original_locations(error, original, edits, source):
    for finding in error.findings:
        finding["source"] = finding.get("source") or source
        if "byte_offset" in finding:
            offset = _map_to_original(finding["byte_offset"], edits)
            location = _finding("", "", offset=offset, data=original)
            finding.update({key: location[key] for key in ("line", "column", "byte_offset")})
    return error


def _namespace_inspection(data, *, context=None, source=""):
    scopes, root = [], None
    missing, declarations, uses = [], [], []

    def start(name, attrs, offset):
        nonlocal root
        scope = dict(scopes[-1]) if scopes else {"xml": XML_NS}
        for key, value in attrs.items():
            if key == "xmlns":
                scope[""] = value
            elif key.startswith("xmlns:"):
                prefix = key[6:]
                scope[prefix] = value
                if prefix == "xsi":
                    declarations.append(offset)
        scopes.append(scope)

        def resolve(qname, attribute=False):
            if ":" in qname:
                prefix, local = qname.split(":", 1)
                return scope.get(prefix), local, prefix
            return ("" if attribute else scope.get("", "")), qname, ""

        uri, local, prefix = resolve(name)
        if root is None:
            root = (uri, local, offset)
        if prefix and uri is None:
            _fail("missing_geometry_namespace", f"The element prefix '{prefix}' is not declared.", source, action="Restore the correct namespace declarations in the original GIS export; do not remove prefixes.", offset=offset, data=data)
        if uri == "http://www.w3.org/2000/09/xmldsig#" or (uri == "http://www.w3.org/2001/XInclude"):
            _fail("unsupported_signed_or_included_xml", "Signed XML and XInclude documents are outside automatic repair.", source)
        for key, value in attrs.items():
            if key == "xmlns" or key.startswith("xmlns:"):
                continue
            attr_uri, attr_local, attr_prefix = resolve(key, True)
            if attr_prefix == "xsi":
                uses.append((uri, local, attr_local, offset))
            if attr_prefix and attr_uri is None:
                if attr_prefix != "xsi" or uri != KML_NS or local not in {"Document", "Folder"} or attr_local not in {"schemaLocation", "noNamespaceSchemaLocation"}:
                    _fail("unsupported_unbound_prefix", f"The attribute '{key}' has an unsupported missing namespace.", source, offset=offset, data=data)
                missing.append(offset)
            if "xsi:" in value and not key.startswith("xsi:"):
                _fail("ambiguous_qname", "A metadata value may depend on the missing xsi namespace.", source, offset=offset, data=data)

    def declaration(version, encoding, standalone):
        if version != "1.0" or (encoding and encoding.lower().replace("-", "") != "utf8"):
            _fail("unsupported_repair_encoding", "Automatic repair requires an XML 1.0 UTF-8 declaration consistent with the bytes.", source)

    _preflight(data, context=context, start=start, end=lambda *_: scopes.pop(), declaration=declaration)
    if root is None or root[:2] != (KML_NS, "kml"):
        _fail("missing_kml_namespace", "Automatic repair requires the explicit KML 2.2 root namespace.", source, action="Re-export complete KML with the correct KML namespace; do not guess missing geometry bindings.")
    if missing:
        if declarations or any(uri != KML_NS or local not in {"Document", "Folder"} or attribute not in {"schemaLocation", "noNamespaceSchemaLocation"} for uri, local, attribute, _ in uses):
            _fail("conflicting_xsi_namespace", "The xsi namespace is declared or used inconsistently and cannot be inserted safely.", source)
    return root, bool(missing)


def _root_insertion(data, token):
    index = token.start_end - 1
    if token.empty:
        index -= 1
        while data[index:index + 1] in (b" ", b"\t", b"\r", b"\n"):
            index -= 1
        if data[index:index + 1] != b"/":
            _fail("invalid_root_tag", "The root start tag cannot be located reliably.", category="verification")
    return index


def _inspect_document(data: bytes, source: str, *, context=None) -> DocumentRepair:
    """Return one proven candidate, or decline without guessing or changing input."""
    try:
        safe_xml_root(data, context=context)
        return DocumentRepair(source, data, data)
    except ET.ParseError:
        pass
    _utf8_document(data, source)
    edits = []
    leading = _leading_patch(data)
    if leading:
        edits.append(leading)
    tokens = _scan(data, context=context)
    for token in tokens:
        if not token.plain or token.empty or _local(token.name) not in _LEAVES or token.parent is None or _local(tokens[token.parent].name) not in _CONTAINERS:
            continue
        position = token.start_end
        while True:
            position = data.find(b"&", position, token.close_start)
            if position < 0:
                break
            if position + 1 == token.close_start or data[position + 1] in _SPACE:
                edits.append(RepairEdit("literal_metadata_ampersand_v1", position, b"&", b"&amp;"))
            position += 1
            if len(edits) > MAX_PATCHES:
                _fail("repair_patch_limit", "The document needs more formatting edits than this repair version supports.", source, category="limit")
    edits.sort(key=lambda e: e.offset)
    _check_patch_limits(edits, source)
    provisional = _apply(data, edits)
    namespace_checked = False
    try:
        _, missing = _namespace_inspection(provisional, context=context, source=source)
        namespace_checked = True
        if missing:
            root = tokens[0]
            edits.append(RepairEdit("missing_xsi_schema_namespace_v1", _root_insertion(data, root), b"", _DECLARATION))
            edits.sort(key=lambda e: e.offset)
        candidate = _apply(data, edits)
        return verify_document(data, candidate, tuple(edits), source=source, context=context)
    except RepairFailure as error:
        # Only preflight findings refer to the provisional document; verifier
        # errors use original positions or intentionally omit unproven positions.
        if not namespace_checked:
            _original_locations(error, data, edits, source)
        raise
    except ET.ParseError as error:
        line, column = getattr(error, "position", (None, None))
        item = _finding("malformed_xml", f"The XML remains invalid: {error}", source, action="Correct the reported XML formatting in the original GIS export and provide the full file.")
        # Candidate locations are not falsely labelled original locations.
        item["location_note"] = "Parser location refers to the formatting candidate; the original location is not confirmed."
        item["candidate_line"], item["candidate_column"] = line, column
        raise RepairFailure(str(error), findings=[item]) from error


def inspect_document(data: bytes, source: str, *, context=None) -> DocumentRepair:
    """Return one verified formatting candidate; never infer missing geometry."""
    try:
        return _inspect_document(data, source, context=context)
    except RepairFailure as error:
        for finding in error.findings:
            finding["source"] = finding.get("source") or source
        raise


def _decoded_leaf(raw):
    """Decode only XML's fixed references; explicitly eligible bare '&' is text."""
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    result, position = [], 0
    fixed = {"amp": "&", "lt": "<", "gt": ">", "apos": "'", "quot": '"'}
    while position < len(text):
        char = text[position]
        if char != "&":
            result.append(char)
            position += 1
            continue
        if position + 1 == len(text) or text[position + 1] in " \t\r\n":
            result.append("&")
            position += 1
            continue
        end = text.find(";", position + 1)
        if end < 0:
            _fail("unknown_entity", "An XML reference is incomplete.")
        name = text[position + 1:end]
        try:
            if name in fixed:
                value = fixed[name]
            elif name.startswith("#x"):
                value = chr(int(name[2:], 16))
            elif name.startswith("#"):
                value = chr(int(name[1:], 10))
            else:
                raise ValueError
        except (ValueError, OverflowError):
            _fail("unknown_entity", "An entity reference has no unambiguous supported expansion.")
        result.append(value)
        position = end + 1
    return "".join(result)


def verify_document(original: bytes, candidate: bytes, edits, *, source="", context=None) -> DocumentRepair:
    """Independently check locations, contexts, inverse bytes, and protected structure."""
    edits = tuple(edits)
    if any(not isinstance(edit, RepairEdit) for edit in edits):
        _fail("invalid_patch_ledger", "The proposed repair ledger is invalid.", source, category="verification")
    _check_patch_limits(edits, source)
    if tuple(sorted(edits, key=lambda e: e.offset)) != edits or len({e.offset for e in edits}) != len(edits):
        _fail("invalid_patch_ledger", "The proposed patch offsets are ambiguous or unordered.", source, category="verification")
    if _apply(original, edits) != candidate:
        _fail("unrecorded_change", "The proposed document contains an unrecorded change.", source, category="verification")
    root = safe_xml_root(candidate, context=context)
    if not edits:
        return DocumentRepair(source, original, candidate)
    _utf8_document(original, source)
    if root.tag != f"{{{KML_NS}}}kml":
        _fail("missing_kml_namespace", "The repaired document has no explicit KML 2.2 root.", source)
    old_tokens = _scan(original, context=context)
    token_starts = [token.start for token in old_tokens]
    new_tokens = _scan(candidate, context=context)
    elements = list(root.iter())
    if len(new_tokens) != len(elements) or [(t.name, t.parent, t.empty) for t in old_tokens] != [(t.name, t.parent, t.empty) for t in new_tokens]:
        _fail("structure_changed", "The proposed repair changes XML structure or feature membership.", source, category="verification")
    # Reconstruct the original from candidate offsets, independently of forward application.
    inverse, position, delta = [], 0, 0
    namespace_edits = []
    changed_leaves = set()
    for edit in edits:
        _check(context)
        location = edit.offset + delta
        if candidate[location:location + len(edit.inserted)] != edit.inserted:
            _fail("inverse_failed", "The patch ledger cannot recover the original document.", source, category="verification")
        inverse.extend((candidate[position:location], edit.removed))
        position = location + len(edit.inserted)
        delta += len(edit.inserted) - len(edit.removed)
        if edit.rule == "leading_xml_whitespace_v1":
            expected = _leading_patch(original)
            if edit != expected:
                _fail("invalid_whitespace_patch", "The patch does not remove exclusively leading XML whitespace.", source, category="verification")
        elif edit.rule == "missing_xsi_schema_namespace_v1":
            if edit.removed or edit.inserted != _DECLARATION or edit.offset != _root_insertion(original, old_tokens[0]):
                _fail("invalid_namespace_patch", "The patch is not the permitted root namespace insertion.", source, category="verification")
            namespace_edits.append(edit)
        elif edit.rule == "literal_metadata_ampersand_v1":
            if edit.removed != b"&" or edit.inserted != b"&amp;":
                _fail("invalid_metadata_patch", "The patch does not encode exactly one literal ampersand.", source, category="verification")
            eligible = bisect_right(token_starts, edit.offset) - 1
            token = old_tokens[eligible] if eligible >= 0 else None
            if token is None or not token.plain or not token.start_end <= edit.offset < token.close_start:
                _fail("invalid_metadata_context", "A metadata patch targets an attribute or non-leaf content.", source, category="verification")
            element = elements[eligible]
            if token.parent is None or _split(element.tag) not in {(KML_NS, n) for n in _LEAVES} or _split(elements[token.parent].tag) not in {(KML_NS, n) for n in _CONTAINERS}:
                _fail("invalid_metadata_context", "A metadata patch targets an unsupported namespace or element.", source, category="verification")
            if edit.offset + 1 != token.close_start and original[edit.offset + 1] not in _SPACE:
                _fail("ambiguous_ampersand", "The proposed ampersand may be an incomplete entity reference.", source, category="verification")
            changed_leaves.add(eligible)
        else:
            _fail("unknown_repair_rule", "The proposed patch uses an unsupported repair rule.", source, category="verification")
    inverse.append(candidate[position:])
    if b"".join(inverse) != original:
        _fail("inverse_failed", "Inverse repair did not recover the exact source bytes.", source, category="verification")
    # Independent namespace inspection without the declaration being justified.
    without_namespace = _apply(original, tuple(e for e in edits if e not in namespace_edits))
    try:
        _, needs_namespace = _namespace_inspection(without_namespace, context=context, source=source)
    except RepairFailure as error:
        raise _original_locations(error, original, tuple(e for e in edits if e not in namespace_edits), source)
    if len(namespace_edits) != int(needs_namespace):
        _fail("unjustified_namespace_patch", "The namespace insertion has no unique supported justification.", source, category="verification")
    for index in changed_leaves:
        token = old_tokens[index]
        if _decoded_leaf(original[token.start_end:token.close_start]) != (elements[index].text or ""):
            _fail("metadata_changed", "The displayed metadata text would change.", source, category="verification")
    # Every geometry subtree, including unmeasured polygons and points, remains exact.
    for old, new, element in zip(old_tokens, new_tokens, elements):
        uri, local = _split(element.tag)
        if local in _GEOMETRY or (uri == GX_NS and local in {"Track", "MultiTrack"}):
            if original[old.start:old.end] != candidate[new.start:new.end]:
                _fail("geometry_changed", "The proposed repair changes a geometry subtree.", source, category="verification")
    rules = tuple(dict.fromkeys(e.rule for e in edits))
    return DocumentRepair(source, original, candidate, edits, rules)


def inspect_geometry(data_or_root, *, source="", context=None):
    """Independent, ordered projection for the app's supported line/point model.

    Names remain empty where the application uses a graph-wide generated name.
    No mutation or omission is concealed: findings block repaired-source coverage.
    """
    root = safe_xml_root(data_or_root, context=context) if isinstance(data_or_root, bytes) else data_or_root
    result = {"pipelines": [], "points": [], "features": [], "findings": [], "counts": {"placemarks": 0, "paths": 0, "vertices": 0, "element_count": 0}}
    parents = {child: parent for parent in root.iter() for child in parent}
    reserved = {name.lower(): name for name in _RESERVED}
    gx_reserved = {name.lower(): name for name in _GX_RESERVED}

    def issue(code, message, ordinal=None, **details):
        if len(result["findings"]) >= MAX_FINDINGS:
            return
        entry = _finding(code, message, source, action="Correct this issue in the original GIS dataset and re-export the complete system without deleting or reconnecting geometry.")
        if ordinal:
            entry["feature_ordinal"] = ordinal
        entry.update(details)
        result["findings"].append(entry)

    for number, element in enumerate(root.iter()):
        result["counts"]["element_count"] += 1
        if number % 128 == 0:
            _check(context)
        uri, local = _split(element.tag)
        if uri in {"http://www.w3.org/2001/XInclude", "http://www.w3.org/2000/09/xmldsig#"}:
            issue("unsupported_signed_or_included_xml", "This document contains signed XML or XInclude references; a complete unchanged source cannot be certified without interpreting unsupported dependencies.")
        if (uri, local) == (KML_NS, "Update"):
            issue("unsupported_update_geometry", "A KML Update contains changes to another document, rather than an independently complete static system; the effective geometry cannot be verified.")
        expected = reserved.get(local.lower()) or gx_reserved.get(local.lower())
        if expected and local != expected:
            issue("geometry_case_ambiguity", f"Element '{local}' resembles '{expected}' but has different case; its geometry cannot be interpreted reliably.")
        if local in _RESERVED and uri != KML_NS:
            issue("geometry_namespace_ambiguity", f"Element '{local}' uses an unsupported geometry/feature namespace.")
        if local in _GX_RESERVED and uri != GX_NS:
            issue("geometry_namespace_ambiguity", f"Element '{local}' uses an unsupported track namespace.")
        parent = parents.get(element)
        parent_tag = _split(parent.tag) if parent is not None else ("", "")
        if local == "coordinates" and parent_tag not in {(KML_NS, "LineString"), (KML_NS, "LinearRing"), (KML_NS, "Point")}:
            issue("unknown_coordinate_structure", "A coordinate element occurs outside a recognized geometry structure.")
        if (local == "coordinates" or (uri, local) == (GX_NS, "coord")) and len(element):
            issue("unknown_coordinate_structure", "A coordinate element contains nested markup; some coordinate content could be ignored.")
        if (uri, local) in {(KML_NS, "LineString"), (KML_NS, "Point"), (KML_NS, "LinearRing")}:
            if sum(_local(child.tag) == "coordinates" for child in element) > 1:
                issue("ambiguous_coordinate_structure", f"Geometry '{local}' contains multiple coordinate elements; the intended complete path is ambiguous.")
        if (uri, local) == (GX_NS, "coord") and parent_tag != (GX_NS, "Track"):
            issue("unknown_coordinate_structure", "A track coordinate occurs outside gx:Track.")
        if uri == KML_NS and local in {"Document", "Folder", "NetworkLink", "Placemark"}:
            if parent_tag not in {(KML_NS, "kml"), (KML_NS, "Document"), (KML_NS, "Folder")}:
                issue("ambiguous_feature_membership", f"Feature '{local}' occurs outside a supported document or folder position.")
        if (uri, local) == (KML_NS, "NetworkLink"):
            links = [child for child in element if _local(child.tag).lower() in {"link", "url"}]
            if len(links) != 1 or _split(links[0].tag) not in {(KML_NS, "Link"), (KML_NS, "Url")}:
                issue("ambiguous_network_link", "A NetworkLink does not contain exactly one correctly named KML Link or Url; its intended source cannot be verified.")
            else:
                hrefs = [child for child in links[0] if _local(child.tag).lower() == "href"]
                if len(hrefs) != 1 or hrefs[0].tag != f"{{{KML_NS}}}href" or len(hrefs[0]):
                    issue("ambiguous_network_link", "A NetworkLink href has ambiguous case, namespace, multiplicity, or nested content; its complete target cannot be verified.")
        if (uri, local) == (KML_NS, "Placemark"):
            ancestor = parent
            while ancestor is not None:
                if _split(ancestor.tag) == (KML_NS, "Placemark"):
                    issue("nested_placemark", "Nested Placemarks have ambiguous feature membership.")
                    break
                ancestor = parents.get(ancestor)
        if local in (_GEOMETRY - {"LinearRing"}) or (uri == GX_NS and local in {"Track", "MultiTrack"}):
            if parent_tag not in {(KML_NS, "Placemark"), (KML_NS, "MultiGeometry"), (GX_NS, "MultiTrack")}:
                issue("ambiguous_geometry_membership", f"Geometry '{local}' occurs outside a supported feature/container position.")

    def direct(element, name):
        return next((child for child in element if _local(child.tag) == name), None)

    def text(element):
        return (element.text or "").strip() if element is not None else ""

    def coordinates(element, ordinal, *, track=False, path_index=None):
        geometry = "gx:Track" if track else _local(element.tag)
        if track:
            values = [text(e).split() for e in element.iter() if e.tag == f"{{{GX_NS}}}coord"]
        else:
            raw = text(direct(element, "coordinates"))
            values = [value.split(",") for value in raw.split()]
        path = []
        for index, value in enumerate(values):
            if index % 256 == 0:
                _check(context)
            try:
                if not 2 <= len(value) <= 3:
                    raise ValueError
                lon, lat = float(value[0]), float(value[1])
                if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
                    raise ValueError
                if len(value) == 3 and not math.isfinite(float(value[2])):
                    raise ValueError
                path.append((lon, lat))
            except (ValueError, IndexError):
                details = {"geometry": geometry, "tuple_index": index + 1}
                if path_index is not None:
                    details["path_index"] = path_index
                issue("invalid_gx_coord" if track else "invalid_coordinate", "A geometry contains an invalid, missing, non-finite, or out-of-range coordinate, altitude, or tuple field. Its vertices cannot be guessed or removed.", ordinal, **details)
                return [], False
        return path, True

    for ordinal, placemark in enumerate((e for e in root.iter() if e.tag == f"{{{KML_NS}}}Placemark"), 1):
        result["counts"]["placemarks"] += 1
        _check(context)
        name, objectid = text(direct(placemark, "name")), "N/A"
        for element in placemark.iter():
            if _local(element.tag) == "Data" and element.get("name") == "OBJECTID":
                value = text(direct(element, "value"))
                if value:
                    objectid = value
                    break
            elif _local(element.tag) == "SimpleData" and element.get("name") == "OBJECTID" and text(element):
                objectid = text(element)
                break
        line_paths, track_paths = [], []
        line_index = track_index = 0
        first_point_valid = False
        first_point_seen = False
        for element in placemark.iter():
            if element.tag == f"{{{KML_NS}}}LineString":
                line_index += 1
                path, valid = coordinates(element, ordinal, path_index=line_index)
                if len(path) >= 2:
                    line_paths.append(path)
                elif valid:
                    issue("short_linestring", "A LineString contains fewer than two coordinates.", ordinal, geometry="LineString", path_index=line_index)
            elif element.tag == f"{{{GX_NS}}}Track":
                track_index += 1
                path, valid = coordinates(element, ordinal, track=True, path_index=track_index)
                if len(path) >= 2:
                    track_paths.append(path)
                elif valid:
                    issue("short_gx_track", "A gx:Track contains fewer than two coordinates.", ordinal, geometry="gx:Track", path_index=track_index)
            elif element.tag in {f"{{{KML_NS}}}Point", f"{{{KML_NS}}}LinearRing"}:
                path, valid = coordinates(element, ordinal)  # Preserve, but never conceal damaged unmeasured geometry.
                if element.tag == f"{{{KML_NS}}}Point":
                    if not first_point_seen:
                        first_point_valid = bool(path) and valid
                        first_point_seen = True
                    if valid and not path:
                        issue("missing_point_coordinate", "A Point has no coordinate tuple.", ordinal, geometry="Point")
                    elif valid and len(path) != 1:
                        issue("ambiguous_point_coordinate", "A Point contains multiple coordinate tuples; its geometry semantics are ambiguous.", ordinal, geometry="Point")
        paths = line_paths + track_paths
        common = {"name": name, "objectid": objectid, "feature_ordinal": ordinal}
        if paths:
            record = dict(common, placemark_id=(placemark.get("id") or "").strip() or "N/A", coordinate_paths=paths)
            result["pipelines"].append(record)
            result["features"].append(dict(record, kind="pipeline"))
            result["counts"]["paths"] += len(paths)
            result["counts"]["vertices"] += sum(map(len, paths))
        else:
            if first_point_valid:
                result["points"].append(common)
                result["features"].append(dict(common, kind="point"))
            elif not (line_index or track_index or first_point_seen) and not any(_local(e.tag) in {"Polygon", "LinearRing", "Model"} for e in placemark.iter()):
                issue("no_supported_geometry", "A Placemark contains no supported pipeline or point geometry.", ordinal)
    if len(result["findings"]) == MAX_FINDINGS:
        result["findings"][-1] = _finding("inspection_findings_truncated", "Further findings were not collected because the inspection limit was reached.", source, category="limit")
    return result
