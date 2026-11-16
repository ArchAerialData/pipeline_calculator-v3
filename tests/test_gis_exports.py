"""Source layer identity survives mileage/state analysis and saved reports."""
from copy import deepcopy
import json
from xml.etree import ElementTree as ET

from openpyxl import load_workbook
from shapely.geometry import box

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.geography.boundaries import BoundaryDataset
from pipeline_calculator.core.placemarks import copy_source_provenance, summarize_placemarks
from pipeline_calculator.core.state_analysis import build_state_breakdown
from pipeline_calculator.export.geography_kmz import build_geography_kml
from pipeline_calculator.export.package import export_analysis_package
from pipeline_calculator.export.xlsx import build_analysis_workbook
from pipeline_calculator.parsers.kml_kmz import ParseResult


def source():
    return {
        'source_dataset': '=customer.gpkg', 'source_layer': '=Pipelines',
        'source_feature_id': '9223372036854775806', 'source_component_path': [0],
        'source_format': 'GPKG', 'source_crs': 'EPSG:4326', 'source_dimensions': ['XY'],
        'internal_id': 'snapshot:layer:feature',
        'source_record_index': 0, 'source_original_feature_id': 'client-feature',
        'path_component_paths': [[0]],
        'source_coordinate_operations': [{'component_path': [0], 'operation_id': 'identity'}],
        'attributes': {'asset_id': 9223372036854775806, 'tags': ['gas']},
    }


def result_with_states():
    provenance = source()
    path = [(-0.001, 0.001), (0.001, 0.001)]
    parsed = ParseResult(pipelines=[{
        'id': 0, 'placemark_id': 'client-id', 'name': '=Asset',
        'coordinates': path, 'coordinate_paths': [path], **deepcopy(provenance),
    }], placemarks=[{
        'Placemark_ID': 'pin-1', 'Name': '=Pin', 'Count': 1,
        'coordinates': (0, 0), **deepcopy(provenance),
    }])
    parsed.parsed_sources = ['=customer.gpkg::=Pipelines']
    parsed.import_metadata = {'schema_version': 1, 'measurement': 'horizontal',
                              'selected_layers': ['=Pipelines'], 'coverage': {'null': 0}}
    analyzer = PipelineAnalyzer()
    combined = analyzer.analyze_parsed(parsed)
    boundaries = BoundaryDataset({'AA': box(-1, -1, 0, 1), 'BB': box(0, -1, 1, 1)},
                                 {'AA': 'West', 'BB': 'East'})
    combined['geography'] = build_state_breakdown(analyzer, parsed.pipelines, combined, boundaries=boundaries)
    assert combined['geography']['status'] == 'complete'
    return combined


def test_source_copy_detaches_attributes_and_component_paths():
    original = source()
    copied = copy_source_provenance(original)
    copied['attributes']['tags'].append('water')
    copied['source_component_path'].append(9)
    assert original['attributes']['tags'] == ['gas']
    assert original['source_component_path'] == [0]
    assert copied['attributes']['asset_id'] == 9223372036854775806


def test_pin_inventory_groups_distinct_datasets_without_fabricating_kml_folders():
    pins = [dict(source()), dict(source()), dict(source(), source_dataset='other.gpkg')]
    inventory = summarize_placemarks(pins)
    assert inventory['total'] == 3
    assert inventory['folder_count'] == 0
    assert inventory['layer_count'] == 2
    assert sorted(group['count'] for group in inventory['groups']) == [1, 2]
    assert all(group['grouping_kind'] == 'layer' and group['folder_path'] == []
               for group in inventory['groups'])


def test_source_identity_survives_state_fragments_and_metadata_is_detached():
    results = result_with_states()
    rows = results['pipelines'] + results['geography']['fragments']
    rows += [row for state in results['geography']['states'] for row in state['pipelines']]
    for row in rows:
        for key, value in source().items():
            assert row[key] == value
    for state in results['geography']['states']:
        assert state['parsed_sources'] == results['parsed_sources']
        assert state['import_metadata'] == results['import_metadata']
        assert state['import_metadata'] is not results['import_metadata']


def test_saved_workbook_and_json_retain_source_columns_as_text(tmp_path):
    results = result_with_states()
    before = deepcopy(results)
    package = export_analysis_package(results, tmp_path, 'pipes.gpkg', include_maps=False, include_json=True)
    workbook = load_workbook(package / 'analysis.xlsx')
    for sheet_name, header_row, data_row in [('Pipeline Length Analysis', 1, 2),
                                            ('Point Pins', 7, 8), ('State Pipeline Lengths', 1, 2)]:
        sheet = workbook[sheet_name]
        headers = [cell.value for cell in sheet[header_row]]
        for header, expected in [('Source Dataset', '=customer.gpkg'), ('Source Layer', '=Pipelines'),
                                 ('Source Feature ID', '9223372036854775806')]:
            cell = sheet.cell(data_row, headers.index(header) + 1)
            assert cell.value == expected
            assert cell.data_type == 's'
        assert sheet.cell(data_row, headers.index('Subfolder') + 1).value is None
        assert sheet.cell(data_row, headers.index('Source KML') + 1).value is None
    assert 'Import Details' in workbook
    assert workbook['Pipeline Length Analysis']['D2'].data_type == 'f'
    exported = json.loads((package / 'analysis.json').read_text(encoding='utf-8'))
    assert exported['import_metadata'] == results['import_metadata']
    assert exported['pipelines'][0]['attributes']['asset_id'] == 9223372036854775806
    assert results == before


def test_kmz_source_metadata_contains_original_layer_and_import_decisions():
    results = result_with_states()
    root = ET.fromstring(build_geography_kml(results))
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    values = [node.text for node in root.findall('.//k:Data[@name="source_dataset"]/k:value', ns)]
    assert values == ['=customer.gpkg', '=customer.gpkg']
    metadata = root.find('.//k:Data[@name="import_metadata"]/k:value', ns)
    assert json.loads(metadata.text) == results['import_metadata']


def test_legacy_workbook_keeps_existing_columns():
    workbook = build_analysis_workbook({'pipelines': [], 'placemarks': []})
    assert [cell.value for cell in workbook['Pipeline Length Analysis'][1]][4:] == [
        'Subfolder', 'Folder Path', 'Source KML']
    assert 'Import Details' not in workbook
