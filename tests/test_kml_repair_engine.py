import json
import xml.etree.ElementTree as ET

import pytest

from pipeline_calculator.core.execution import AnalysisCancelled, ExecutionContext
from pipeline_calculator.parsers import repair
from pipeline_calculator.parsers.kml_kmz import _ParserState, _parse_kml_bytes


KML = repair.KML_NS
GX = repair.GX_NS
XSI = repair.XSI_NS
DECLARATION = f' xmlns:xsi="{XSI}"'.encode()
LINE = b'<LineString><coordinates>-101,35,42 -100,36,43 -100,36,43</coordinates></LineString>'


def document(body=LINE, *, metadata=b'', attribute=b'', namespaces=b'', declaration=b''):
    return declaration + b'<kml xmlns="' + KML.encode() + b'"' + namespaces + b'><Document' + attribute + b'>' + metadata + b'<Placemark id="same"><name>Pipe</name>' + body + b'</Placemark></Document></kml>'


def parse(data):
    state = _ParserState()
    _parse_kml_bytes(data, state, source='doc.kml', required=True)
    return state


def test_missing_xsi_only_inserts_one_declaration_and_is_idempotent():
    original = document(attribute=b' xsi:schemaLocation="http://example.invalid/schema"')
    # Independent complete root; not produced by the repair under test.
    authored = (f'<kml xmlns="{KML}" xmlns:xsi="{XSI}"><Document xsi:schemaLocation="http://example.invalid/schema"><Placemark id="same"><name>Pipe</name>'.encode() + LINE + b'</Placemark></Document></kml>')
    result = repair.inspect_document(original, 'doc.kml')
    assert result.rules == ('missing_xsi_schema_namespace_v1',)
    assert len(result.edits) == 1
    edit = result.edits[0]
    assert edit.inserted == DECLARATION and not edit.removed
    assert original[:edit.offset] + edit.inserted + original[edit.offset:] == result.effective
    assert parse(result.effective).pipelines == parse(authored).pipelines
    assert repair.inspect_document(result.effective, 'doc.kml').edits == ()
    assert json.loads(json.dumps(result.to_report()))['byte_preservation'] == 'passed'


@pytest.mark.parametrize('prefix', [b' ', b'\t\r\n  ', b'\xef\xbb\xbf \n\t'])
def test_whitespace_preserves_bom_declaration_and_entire_document(prefix):
    original = prefix + document(declaration=b'<?xml version="1.0" encoding="UTF-8"?>')
    result = repair.inspect_document(original, 'leading.kml')
    expected = (b'\xef\xbb\xbf' if prefix.startswith(b'\xef\xbb\xbf') else b'') + document(declaration=b'<?xml version="1.0" encoding="UTF-8"?>')
    assert result.effective == expected
    assert result.rules == ('leading_xml_whitespace_v1',)


@pytest.mark.parametrize('container', ['Document', 'Folder', 'Placemark'])
@pytest.mark.parametrize('leaf', ['name', 'description', 'Snippet'])
@pytest.mark.parametrize('value,expected', [(b'A & B', 'A & B'), (b'End &', 'End &'), (b'&amp; &\r\n &#13; &lt;', '& &\n \r <')])
def test_literal_metadata_preserves_exact_decoded_text(container, leaf, value, expected):
    content = f'<{container}><{leaf}>'.encode() + value + f'</{leaf}></{container}>'.encode()
    original = document(metadata=content)
    result = repair.inspect_document(original, 'metadata.kml')
    root = ET.fromstring(result.effective)
    element = root.find(f'.//{{{KML}}}{container}/{{{KML}}}{leaf}')
    assert element is not None and element.text == expected
    assert all(edit.rule == 'literal_metadata_ampersand_v1' for edit in result.edits)
    assert LINE in result.effective


def test_composed_rules_use_original_multibyte_offsets_and_keep_all_geometry():
    mixed = (b'<MultiGeometry><gx:Track><when>2026-01-01T00:00:00Z</when><gx:coord>-110 20 100</gx:coord><gx:coord>-109 21 200</gx:coord></gx:Track>'
             + LINE + b'<Point><coordinates>-100,35,10</coordinates></Point><Polygon><outerBoundaryIs><LinearRing><coordinates>-102,34,1 -101,34,2 -101,35,3 -102,34,1</coordinates></LinearRing></outerBoundaryIs></Polygon></MultiGeometry>')
    original = b'\xef\xbb\xbf\r\n ' + document(mixed, metadata='<description>Café & route</description>'.encode(), attribute=b' xsi:schemaLocation="a b"', namespaces=f' xmlns:gx="{GX}"'.encode(), declaration=b'<?xml version="1.0" encoding="UTF-8"?>')
    result = repair.inspect_document(original, 'mixed.kml')
    assert set(result.rules) == {'leading_xml_whitespace_v1', 'missing_xsi_schema_namespace_v1', 'literal_metadata_ampersand_v1'}
    assert original[result.edits[-1].offset:result.edits[-1].offset + 1] == b'&'
    assert mixed in result.effective
    inventory = repair.inspect_geometry(result.effective)
    assert not inventory['findings']
    assert inventory['pipelines'][0]['coordinate_paths'] == [
        [(-101., 35.), (-100., 36.), (-100., 36.)],
        [(-110., 20.), (-109., 21.)],
    ]
    assert parse(result.effective).pipelines[0]['coordinate_paths'] == inventory['pipelines'][0]['coordinate_paths']


@pytest.mark.parametrize('metadata', [
    b'<name>A&B</name>', b'<name>&nbsp;</name>', b'<name>&amp</name>',
    b'<name>&#xno;</name>', b'<name>A < B</name>',
    b'<name>A & <b>B</b></name>', b'<name>A & <!-- comment --> B</name>',
    b'<name>A & <![CDATA[B]]></name>', b'<name>A & <?test x?>B</name>',
    b'<ExtendedData><name>A & B</name></ExtendedData>',
    b'<name title="A & B">OK</name>', b'<name xmlns="urn:foreign">A & B</name>',
    b'<other:name xmlns:other="urn:foreign">A & B</other:name>',
])
def test_near_miss_ampersands_refused(metadata):
    with pytest.raises(repair.RepairFailure):
        repair.inspect_document(document(metadata=metadata), 'unsafe.kml')


@pytest.mark.parametrize('body', [
    b'<LineString><coordinates>-101,35 & -100,36</coordinates></LineString>',
    b'<NetworkLink><Link><href>A & B.kml</href></Link></NetworkLink>',
    b'<gx:Track><gx:coord>-101 35 &</gx:coord></gx:Track>',
])
def test_geometry_links_and_tracks_are_never_escaped(body):
    with pytest.raises(repair.RepairFailure):
        repair.inspect_document(document(body, namespaces=f' xmlns:gx="{GX}"'.encode()), 'unsafe.kml')


@pytest.mark.parametrize('attribute', [b' xsi:type="LineString"', b' xsi:nil="true"', b' foo:kind="x"'])
def test_unbound_non_schema_attribute_refused(attribute):
    with pytest.raises(repair.RepairFailure):
        repair.inspect_document(document(attribute=attribute), 'unsafe.kml')


@pytest.mark.parametrize('body', [
    b'<gx:Track><gx:coord>-1 1</gx:coord><gx:coord>-2 2</gx:coord></gx:Track>',
    b'<xsi:Document/>', b'<LineString xsi:schemaLocation="a b"/>',
])
def test_unbound_geometry_namespace_or_geometry_attributes_refused(body):
    with pytest.raises(repair.RepairFailure):
        repair.inspect_document(document(body, attribute=b' xsi:schemaLocation="a b"'), 'unsafe.kml')


def test_nested_xsi_declaration_does_not_justify_changing_outer_scope():
    body = f'<Folder xmlns:xsi="{XSI}"/><Folder xsi:schemaLocation="a b"/>'.encode()
    with pytest.raises(repair.RepairFailure, match='inconsistently'):
        repair.inspect_document(document(metadata=body, attribute=b' xsi:schemaLocation="a b"'), 'unsafe.kml')


def test_namespace_uri_references_are_resolved_but_never_normalized():
    original = document(attribute=b' xsi:schemaLocation="a b"').replace(KML.encode(), b'http://www.opengis.net/kml/2.&#50;')
    assert repair.inspect_document(original, 'doc.kml').rules == ('missing_xsi_schema_namespace_v1',)
    with pytest.raises(repair.RepairFailure):
        repair.inspect_document(original.replace(b'2.&#50;', b'2.2/'), 'doc.kml')


@pytest.mark.parametrize('prefix', [b'<!-- banner -->', b'junk', b'\xef\xbb\xbf\xef\xbb\xbf', b'\xc2\xa0', b'<?work done?>', b'<?xml version="1.0"?>'])
def test_arbitrary_prefix_is_not_trimmed(prefix):
    with pytest.raises(repair.RepairFailure):
        repair.inspect_document(prefix + document(declaration=b'<?xml version="1.0"?>'), 'unsafe.kml')


@pytest.mark.parametrize('change', ['truncate', 'root', 'case_namespace', 'bad_utf8', 'wrong_encoding', 'signed'])
def test_composed_repair_fails_as_a_transaction_on_other_defects(change):
    original = document(metadata=b'<name>A & B</name>', attribute=b' xsi:schemaLocation="a b"')
    if change == 'truncate':
        original = original[:-6]
    elif change == 'root':
        original += b'<kml/>'
    elif change == 'case_namespace':
        original = original.replace(KML.encode(), KML.upper().encode())
    elif change == 'bad_utf8':
        original = original.replace(b'A & B', b'\xff & B')
    elif change == 'wrong_encoding':
        original = b'<?xml version="1.0" encoding="iso-8859-1"?>' + original
    elif change == 'signed':
        original = original.replace(b'</Document>', b'<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"/></Document>')
    with pytest.raises(repair.RepairFailure):
        repair.inspect_document(original, 'unsafe.kml')


@pytest.mark.parametrize('valid', [
    document(), b' \r\n' + document() + b'\r\n ',
    b'\xef\xbb\xbf' + document(declaration=b'<?xml version="1.0"?>'),
    document(metadata=b'<description><![CDATA[A & B <tag>]]></description><!-- xsi:schemaLocation="x" -->'),
    document(metadata=b'<name>A &amp; B &#38; C</name>'),
    document(declaration=b'<?xml version="1.0" encoding="UTF-16"?>').decode().encode('utf-16'),
])
def test_valid_controls_are_identical_and_need_no_repair(valid):
    result = repair.inspect_document(valid, 'valid.kml')
    assert result.original == result.effective == valid and result.edits == ()


@pytest.mark.parametrize('declaration', [
    b'<!DOCTYPE kml [<!ENTITY name "replacement">]>',
    b'<!DOCTYPE kml SYSTEM "file:///do-not-access.dtd">',
    b'<!DOCTYPE kml [<!ENTITY ext SYSTEM "https://example.invalid/">]>',
])
def test_dtd_rejected_before_tree_parse_or_expansion(declaration, monkeypatch):
    called = []
    monkeypatch.setattr(repair.ET, 'fromstring', lambda *_: called.append(True))
    with pytest.raises(repair.RepairFailure, match='DTD'):
        repair.safe_xml_root(declaration + document())
    assert called == []


def test_comment_and_cdata_dtd_lookalikes_are_allowed():
    data = document(metadata=b'<!-- <!DOCTYPE kml> --><description><![CDATA[<!ENTITY x>]]></description>')
    assert repair.safe_xml_root(data).tag == f'{{{KML}}}kml'


@pytest.mark.parametrize('limit,value,data', [
    ('MAX_XML_BYTES', 12, document()),
    ('MAX_DEPTH', 2, document()),
    ('MAX_ELEMENTS', 3, document()),
    ('MAX_ATTRIBUTES', 1, document(namespaces=f' xmlns:gx="{GX}"'.encode())),
])
def test_safety_limits_apply_before_tree_construction(limit, value, data, monkeypatch):
    monkeypatch.setattr(repair, limit, value)
    monkeypatch.setattr(repair.ET, 'fromstring', lambda *_: pytest.fail('tree constructed before bounds'))
    with pytest.raises(repair.RepairFailure) as caught:
        repair.safe_xml_root(data)
    assert caught.value.category == 'limit'
    assert 'Contact support' in caught.value.client_request()


def test_cancel_remains_control_flow():
    context = ExecutionContext()
    context.cancel()
    with pytest.raises(AnalysisCancelled):
        repair.inspect_document(document(), 'cancel.kml', context=context)


def test_cancellation_during_scan_is_cooperative():
    class CancelLater:
        checks = 0

        def check(self):
            self.checks += 1
            if self.checks == 8:
                raise AnalysisCancelled()

    context = CancelLater()
    data = document(metadata=b'<name>A & B</name>' * 1000)
    with pytest.raises(AnalysisCancelled):
        repair.inspect_document(data, 'cancel.kml', context=context)


@pytest.mark.parametrize('target,inserted', [
    (b'-101,35,42', b'-102,35,42'), (b'<name>Pipe</name>', b'<name>Other</name>'),
    (b'id="same"', b'id="different"'),
])
def test_verifier_rejects_unauthorized_changes_even_with_forged_rule(target, inserted):
    original = document(attribute=b' xsi:schemaLocation="a b"')
    good = repair.inspect_document(original, 'doc.kml')
    patch = repair.RepairEdit('literal_metadata_ampersand_v1', original.index(target), target, inserted)
    edits = tuple(sorted((*good.edits, patch), key=lambda e: e.offset))
    candidate = repair._apply(original, edits)
    with pytest.raises(repair.RepairFailure):
        repair.verify_document(original, candidate, edits, source='doc.kml')


def test_verifier_rejects_unrecorded_geometry_changes_and_duplicate_offsets():
    original = document(attribute=b' xsi:schemaLocation="a b"')
    good = repair.inspect_document(original, 'doc.kml')
    with pytest.raises(repair.RepairFailure, match='unrecorded'):
        repair.verify_document(original, good.effective.replace(b'-101,35,42', b'-111,35,42'), good.edits)
    with pytest.raises(repair.RepairFailure, match='ambiguous'):
        repair.verify_document(original, good.effective, good.edits * 2)


@pytest.mark.parametrize('geometry,expected', [
    (b'<linestring><coordinates>-101,35 -100,36</coordinates></linestring>', 'geometry_case_ambiguity'),
    (b'<LineString xmlns="urn:foreign"><coordinates>-101,35 -100,36</coordinates></LineString>', 'geometry_namespace_ambiguity'),
    (b'<Thing><coordinates>-101,35 -100,36</coordinates></Thing>', 'unknown_coordinate_structure'),
    (b'<Placemark>' + LINE + b'</Placemark>', 'nested_placemark'),
    (b'<ExtendedData>' + LINE + b'</ExtendedData>', 'ambiguous_geometry_membership'),
    (b'<LineString><coordinates>-101,35 NaN,36</coordinates></LineString>', 'invalid_coordinate'),
    (b'<LineString><coordinates>-101,35</coordinates></LineString>', 'short_linestring'),
])
def test_byte_repair_can_pass_while_coverage_blocks_geometry(geometry, expected):
    original = document(geometry, attribute=b' xsi:schemaLocation="a b"')
    result = repair.inspect_document(original, 'damaged.kml')
    assert result.to_report()['byte_preservation'] == 'passed'
    inventory = repair.inspect_geometry(result.effective, source='damaged.kml')
    assert expected in {f['code'] for f in inventory['findings']}


def test_inventory_keeps_duplicate_feature_identity_and_separate_paths():
    original = (f'<kml xmlns="{KML}"><Document xsi:schemaLocation="a b">'.encode()
                + b'<Placemark id="same"><name>Same</name><ExtendedData><Data name="OBJECTID"><value>42</value></Data></ExtendedData><MultiGeometry>' + LINE + LINE + b'</MultiGeometry></Placemark>'
                + b'<Placemark id="same"><name>Same</name>' + LINE + b'</Placemark></Document></kml>')
    inventory = repair.inspect_geometry(repair.inspect_document(original, 'same.kml').effective)
    assert len(inventory['pipelines']) == 2
    assert [len(p['coordinate_paths']) for p in inventory['pipelines']] == [2, 1]
    assert [p['placemark_id'] for p in inventory['pipelines']] == ['same', 'same']
    assert [p['feature_ordinal'] for p in inventory['pipelines']] == [1, 2]
    assert inventory['pipelines'][0]['objectid'] == '42'


def test_client_request_reports_confirmed_issue_without_suggesting_salvage():
    try:
        repair.inspect_document(document(b'<gx:Track/>'), 'nested/doc.kml')
    except repair.RepairFailure as error:
        request = error.client_request()
    assert 'nested/doc.kml' in request
    assert 'prefix' in request
    assert 'preserving all pipeline vertices' in request
    assert 'Do not delete' in request
    assert 'additional issues may remain' in request


@pytest.mark.parametrize('coordinate', [b'-101,35,NaN', b'-101,35,Infinity', b'-101,35,', b'-101,35,1,2', b'-101,35,wrong'])
def test_repaired_coverage_refuses_invalid_supplied_altitude_and_extra_fields(coordinate):
    data = document(b'<LineString><coordinates>-102,35,10 ' + coordinate + b'</coordinates></LineString>', attribute=b' xsi:schemaLocation="a b"')
    inventory = repair.inspect_geometry(repair.inspect_document(data, 'altitude.kml').effective)
    finding = next(f for f in inventory['findings'] if f['code'] == 'invalid_coordinate')
    assert (finding['feature_ordinal'], finding['path_index'], finding['tuple_index']) == (1, 1, 2)
    assert not inventory['pipelines']


def test_invalid_path_does_not_suppress_separate_empty_path_issue():
    body = b'<MultiGeometry><LineString><coordinates>bad 1,2</coordinates></LineString><LineString/></MultiGeometry>'
    inventory = repair.inspect_geometry(document(body))
    findings = [f for f in inventory['findings'] if f['code'] in {'invalid_coordinate', 'short_linestring'}]
    assert [(f['code'], f['path_index']) for f in findings] == [('invalid_coordinate', 1), ('short_linestring', 2)]


@pytest.mark.parametrize('body,code', [
    (b'<LineString><coordinates>-101,35 -100,36</coordinates><coordinates>-99,36 -98,36</coordinates></LineString>', 'ambiguous_coordinate_structure'),
    (b'<LineString><coordinates>-101,35 -100,36<more>-99,36</more></coordinates></LineString>', 'unknown_coordinate_structure'),
    (b'<Point><coordinates>-101,35 -100,36</coordinates></Point>', 'ambiguous_point_coordinate'),
    (b'<Point/>', 'missing_point_coordinate'),
])
def test_potentially_ignored_coordinate_blocks_refuse_coverage(body, code):
    assert code in {f['code'] for f in repair.inspect_geometry(document(body))['findings']}


def test_root_locator_respects_quoted_greater_than_and_actual_namespaces():
    data = (f'<k:kml xmlns:k="{KML}" hint=">"><k:Document xsi:schemaLocation="a b"><k:name>Café & pipeline</k:name><k:Placemark><k:LineString><k:coordinates>-101,35 -100,36</k:coordinates></k:LineString></k:Placemark></k:Document></k:kml>').encode()
    result = repair.inspect_document(data, 'quoted.kml')
    assert result.edits[0].offset == data.index(b'><k:Document')
    assert len(repair.inspect_geometry(result.effective)['pipelines']) == 1


def test_provisional_error_offsets_are_mapped_to_original_bytes():
    data = b' \r\n' + document(declaration=b'<?xml version="1.0"?>', metadata='<name>Café & route</name><gx:Track/>'.encode())
    with pytest.raises(repair.RepairFailure) as caught:
        repair.inspect_document(data, 'bad-prefix.kml')
    finding = caught.value.findings[0]
    assert finding['byte_offset'] == data.index(b'<gx:Track')
    assert finding['line'] == 2
    assert finding['source'] == 'bad-prefix.kml'


@pytest.mark.parametrize('data', [
    document(declaration=b'<?xml version="1.0" encoding="UTF-16"?>').decode().encode('utf-16'),
    document(declaration=b'<?xml version="1.0" encoding="ISO-8859-1"?>'),
    document(declaration=b'<?xml version="1.1" encoding="UTF-8"?>'),
])
def test_repair_encoding_policy_applies_even_to_valid_format_only_corrections(data):
    assert repair.inspect_document(data, 'valid.kml').edits == ()
    with pytest.raises(repair.RepairFailure):
        repair.validate_repair_encoding(data, 'misnamed.kmz')


def test_patch_work_limits_are_not_client_corruption(monkeypatch):
    monkeypatch.setattr(repair, 'MAX_PATCH_BYTES', 2)
    with pytest.raises(repair.RepairFailure) as caught:
        repair.inspect_document(document(metadata=b'<name>A & B</name>'), 'large.kml')
    assert caught.value.category == 'limit'
    assert 'not been shown' in caught.value.client_request()


def test_policy_client_request_requires_an_explicit_corrective_action():
    error = repair.RepairFailure('encrypted', category='policy', findings=[{'category': 'policy', 'code': 'encrypted_zip', 'message': 'Encrypted KMZ is unsupported.', 'action': 'Export the complete source without encryption.'}])
    assert 'Export the complete source without encryption.' in error.client_request()


def test_unknown_encoding_gives_typed_supported_export_feedback():
    data = document(declaration=b'<?xml version="1.0" encoding="not-an-encoding"?>')
    with pytest.raises(repair.RepairFailure) as caught:
        repair.inspect_document(data, 'encoding.kml')
    assert caught.value.category == 'policy'
    assert 'UTF-8' in caught.value.client_request()


def test_original_locations_normalize_cr_and_count_unicode_columns():
    data = b' \r' + document(declaration=b'<?xml version="1.0"?>', metadata='<name>Café & route</name><gx:Track/>'.encode())
    with pytest.raises(repair.RepairFailure) as caught:
        repair.inspect_document(data, 'bad-prefix.kml')
    finding = caught.value.findings[0]
    offset = data.index(b'<gx:Track')
    assert finding['line'] == 2
    assert finding['column'] == len(data[:offset].split(b'\r')[-1].decode('utf-8')) + 1


@pytest.mark.parametrize('content', [
    b'<Link xmlns="urn:foreign"><href>child.kml</href></Link>',
    b'<Link><href xmlns="urn:foreign">child.kml</href></Link>',
    b'<Link><href>child.kml</href><href>another.kml</href></Link>',
    b'<Link><href>child.kml</href></Link><Link><href>another.kml</href></Link>',
    b'<Link><Href>child.kml</Href></Link>',
    b'<link><href>child.kml</href></link>',
    b'<Link><href>child.kml<more>other.kml</more></href></Link>',
])
def test_network_graph_cannot_be_certified_from_ambiguous_link_structure(content):
    source = document(metadata=b'<NetworkLink>' + content + b'</NetworkLink>')
    assert 'ambiguous_network_link' in {f['code'] for f in repair.inspect_geometry(source)['findings']}


@pytest.mark.parametrize('link_name', ['Link', 'Url'])
def test_normal_and_legacy_network_link_structures_remain_supported(link_name):
    source = document(metadata=f'<NetworkLink><{link_name}><href>child.kml</href></{link_name}></NetworkLink>'.encode())
    assert 'ambiguous_network_link' not in {f['code'] for f in repair.inspect_geometry(source)['findings']}


@pytest.mark.parametrize('metadata', [
    b'<xi:include xmlns:xi="http://www.w3.org/2001/XInclude" href="more.kml"/>',
    b'<ds:Signature xmlns:ds="http://www.w3.org/2000/09/xmldsig#"/>',
])
def test_valid_linked_documents_cannot_bypass_unsupported_dependency_coverage(metadata):
    source = document(metadata=metadata)
    assert repair.inspect_document(source, 'linked.kml').edits == ()
    assert 'unsupported_signed_or_included_xml' in {f['code'] for f in repair.inspect_geometry(source)['findings']}


@pytest.mark.parametrize('update', [
    b'<Update><targetHref>other.kml</targetHref><Delete><Placemark targetId="old"/></Delete></Update>',
    b'<Update><targetHref>other.kml</targetHref><Change><Placemark targetId="old">' + LINE + b'</Placemark></Change></Update>',
])
def test_dynamic_update_documents_do_not_claim_static_geometry_completeness(update):
    source = document(metadata=b'<NetworkLinkControl>' + update + b'</NetworkLinkControl>')
    assert 'unsupported_update_geometry' in {f['code'] for f in repair.inspect_geometry(source)['findings']}
