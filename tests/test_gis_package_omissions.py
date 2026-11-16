import json
import zipfile

from pipeline_calculator.importing import prepare_import


def test_generic_zip_reports_unread_kml_and_nested_archives(tmp_path):
    source = tmp_path / 'mixed.zip'
    content = {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'properties': {},
        'geometry': {'type': 'LineString', 'coordinates': [[-100, 30], [-100.001, 30]]}}]}
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('pipes.geojson', json.dumps(content))
        archive.writestr('other.kml', '<kml/>')
        archive.writestr('nested.zip', b'not automatically opened')
        archive.writestr('pins.kmz', b'not automatically opened')
        archive.writestr('pipes.shp.xml', '<opaque/>')
    parsed = prepare_import(source).fresh_parse()
    assert len(parsed.pipelines) == 1
    warnings = [item for item in parsed.diagnostics if item['code'] == 'unsupported_package_members']
    assert len(warnings) == 1 and warnings[0]['level'] == 'warning'
    assert set(warnings[0]['context']['members']) == {'other.kml', 'nested.zip', 'pins.kmz'}
    assert warnings[0]['context']['member_count'] == 3
    assert parsed.import_metadata['selected_input_complete']
