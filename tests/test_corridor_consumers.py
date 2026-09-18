"""Every consumer respects the complete canonical corridor, never a partial ring."""
from copy import deepcopy
import json
import math
import xml.etree.ElementTree as ET

import pytest
from shapely.geometry import Polygon, box

from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml
from pipeline_calculator.export.corridor_metadata import (
    corridor_detail_text, corridor_map_status, validate_corridor_results,
)
from pipeline_calculator.export.geography_kmz import append_corridor, document
from pipeline_calculator.export.xlsx import build_analysis_workbook
from pipeline_calculator.gui.actions.corridor_launch import corridor_is_omitted
from pipeline_calculator.gui.actions.export_actions import export_results_to_path
from test_geography_export import geography_result


NS = {'k': 'http://www.opengis.net/kml/2.2'}


def spec(shape):
    return {'outer': [list(point) for point in shape.exterior.coords],
            'holes': [[list(point) for point in ring.coords] for ring in shape.interiors]}


@pytest.fixture
def buffered_section():
    shape = Polygon([(-100, 30), (-99.99, 30), (-99.99, 30.01), (-100, 30.01)],
                    [[(-99.998, 30.002), (-99.992, 30.002), (-99.992, 30.008), (-99.998, 30.008)]])
    return {
        'pipeline_1': '=A & B', 'pipeline_2': 'C < D', 'pipeline_1_id': 1, 'pipeline_2_id': 2,
        'bundled_length_miles': 1.25, 'bundled_length_meters': 2011.68,
        'average_separation': 12.5, 'segment_count': 402,
        'visualization_schema_version': 1, 'visualization_kind': 'qualified_path_buffer',
        'visualization_status': 'ready', 'visualization_polygons': [spec(shape), spec(box(-99.98, 30, -99.97, 30.01))],
        'visualization_metadata': {'policy': 'qualified_path_buffer_v1', 'padding_m': 5.0,
                                   'cap_style': 'round', 'join_style': 'round', 'source_runs': []},
        'diagnostics': [],
    }


def result_for(section):
    return {'pipelines': [], 'analysis_complete': True, 'total_meters': 10000., 'total_miles': 6.2137,
            'overlap_analysis': {'bundled_sections': [section], 'savings_miles': 1.25}, 'diagnostics': []}


def parsed_polygons(root):
    def coordinates(node):
        return [tuple(map(float, token.split(',')[:2])) for token in node.text.split()]
    return [Polygon(coordinates(p.find('k:outerBoundaryIs/k:LinearRing/k:coordinates', NS)),
                    [coordinates(h) for h in p.findall('k:innerBoundaryIs/k:LinearRing/k:coordinates', NS)])
            for p in root.findall('.//k:Polygon', NS)]


def test_canonical_preview_preserves_all_parts_holes_and_matches_package_serializer(buffered_section):
    before = deepcopy(buffered_section)
    preview = ET.fromstring(build_overlap_corridor_kml(buffered_section, 7))
    root, target = document('Example', 'Example')
    append_corridor(target, buffered_section, 7)
    shapes = parsed_polygons(preview)
    assert len(shapes) == 2 and sum(len(shape.interiors) for shape in shapes) == 1
    assert all(a.equals_exact(b, 0) for a, b in zip(shapes, parsed_polygons(root)))
    assert not preview.findall('.//k:Point', NS) and not preview.findall('.//k:LineString', NS)
    assert len(preview.findall('.//k:Placemark', NS)) == 1
    assert '=A & B' in preview.find('.//k:Placemark/k:name', NS).text
    assert '5 m padding' in preview.find('.//k:Placemark/k:description', NS).text
    assert buffered_section == before


@pytest.mark.parametrize('defect', ['empty', 'omitted', 'invalid', 'future', 'invalid_schema_type'])
def test_canonical_failure_never_resurrects_legacy_bbox(buffered_section, defect):
    buffered_section['bbox'] = {'min_lon': -100, 'max_lon': -99, 'min_lat': 30, 'max_lat': 31}
    if defect == 'empty':
        buffered_section['visualization_polygons'] = []
    elif defect == 'omitted':
        buffered_section.update(visualization_status='omitted', visualization_polygons=[])
    elif defect == 'invalid':
        buffered_section['visualization_polygons'] = [{'outer': [[0, 0], [1, 1], [0, 1], [1, 0]], 'holes': []}]
    elif defect == 'future':
        buffered_section['visualization_schema_version'] = 2
    else:
        buffered_section['visualization_schema_version'] = True
    with pytest.raises(ValueError):
        build_overlap_corridor_kml(buffered_section, 1)
    if defect != 'invalid':
        assert corridor_is_omitted(buffered_section)


def test_clipped_canonical_coordinates_preserve_full_precision(buffered_section):
    clipped = spec(box(-100.12345678901234, 30.12345678901234, -100.12, 30.13))
    buffered_section['clipped_polygons'] = [clipped]
    root = ET.fromstring(build_overlap_corridor_kml(buffered_section, 1))
    assert parsed_polygons(root)[0].equals_exact(Polygon(clipped['outer']), 0)
    assert len(parsed_polygons(root)) == 1  # Never the two uncut polygons.


def test_workbook_adds_map_status_without_changing_numbers_or_leaking_partial_ring(buffered_section, geography_result, tmp_path):
    buffered_section['corridor_polygon'] = buffered_section['visualization_polygons'][0]['outer']
    geography_result['overlap_analysis']['bundled_sections'] = [buffered_section]
    geography_result['geography']['states'][0]['overlap_analysis']['bundled_sections'] = [buffered_section]
    workbook = build_analysis_workbook(geography_result)
    overlap = workbook['Pipeline Overlap Analysis']
    assert [c.value for c in overlap[1]][:4] == ['Pipeline 1', 'Pipeline 2', 'Bundled Length (mi)', 'TOTAL MILEAGE REMOVED']
    assert overlap['C2'].value == 1.25 and overlap['D2'].value == 0
    assert overlap['H2'].value is None and overlap['I2'].value is None
    assert overlap['L2'].value is None and overlap['M2'].value == ''
    assert overlap['N1'].value == 'Corridor Map' and overlap['N2'].value == 'Available (2 parts, 1 hole)'
    assert workbook['State Overlap Analysis']['J2'].value == overlap['N2'].value
    assert workbook.sheetnames.count('Analysis Details') == 1
    assert overlap['A2'].data_type == 's' and overlap['A2'].value == '=A & B'
    assert workbook['Pipeline Length Analysis']['D2'].value == '=SUM(C2:C100000)'
    workbook.save(tmp_path / 'complete.xlsx')


def test_single_ring_excel_overflow_is_omitted_whole(buffered_section, tmp_path):
    ring = [[-100 + .01 * math.cos(i * math.tau / 2000), 30 + .01 * math.sin(i * math.tau / 2000)] for i in range(2000)]
    ring.append(ring[0])
    buffered_section['visualization_polygons'] = [{'outer': ring, 'holes': []}]
    workbook = build_analysis_workbook(result_for(buffered_section))
    assert workbook['Pipeline Overlap Analysis']['M2'].value == ''
    assert '1 part' in workbook['Pipeline Overlap Analysis']['N2'].value
    assert any('30,000 characters' in str(cell.value) for row in workbook['Analysis Details'] for cell in row)
    output = tmp_path / 'analysis.json'
    export_results_to_path(result_for(buffered_section), str(output))
    assert json.loads(output.read_text())['overlap_analysis']['bundled_sections'][0]['visualization_polygons'][0]['outer'] == ring


@pytest.mark.parametrize('bad_value', [object(), float('nan'), float('inf'), (1, 2)])
def test_non_json_canonical_metadata_is_rejected_before_writing(buffered_section, tmp_path, bad_value):
    buffered_section['visualization_metadata']['invalid'] = bad_value
    result = result_for(buffered_section)
    destination = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='JSON-native'):
        export_results_to_path(result, str(destination))
    assert not destination.exists()
    with pytest.raises(ValueError, match='JSON-native'):
        build_analysis_workbook(result)


def test_omitted_geometry_and_metadata_cannot_disagree(buffered_section):
    buffered_section['visualization_status'] = 'omitted'
    with pytest.raises(ValueError, match='omitted corridor'):
        validate_corridor_results(result_for(buffered_section))
    assert corridor_is_omitted(buffered_section)


def test_json_cannot_claim_ready_when_canonical_geometry_is_empty(buffered_section, tmp_path):
    buffered_section['visualization_polygons'] = []
    destination = tmp_path / 'rejected.json'
    with pytest.raises(ValueError, match='ready corridor'):
        export_results_to_path(result_for(buffered_section), str(destination))
    assert not destination.exists()


def test_map_status_and_padding_text_use_real_result_values(buffered_section):
    assert corridor_map_status(buffered_section) == 'Available (2 parts, 1 hole)'
    assert '5 m padding' in corridor_detail_text(result_for(buffered_section))
    buffered_section['visualization_metadata']['padding_m'] = 7.5
    assert '7.5 m padding' in corridor_detail_text(result_for(buffered_section))


@pytest.mark.native_gui
@pytest.mark.parametrize('scale,size', [(1.0, '900x650'), (1.5, '900x650'), (2.0, '900x650'), (1.25, '640x480')])
def test_buffered_map_notes_share_existing_details_and_disable_omitted_action(buffered_section, scale, size):
    import customtkinter as ctk
    from pipeline_calculator.gui.tabs.overlap_tab import create, CorridorTable
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView

    ctk.set_widget_scaling(scale)
    root = ctk.CTk()
    root.geometry(size)
    omitted = dict(buffered_section, visualization_status='omitted', visualization_polygons=[])
    result = result_for(buffered_section)
    result['overlap_analysis']['bundled_sections'].append(omitted)
    opened = []
    try:
        create(root, result, on_open_corridor=lambda *args: opened.append(args))
        root.update()
        table = next(child for child in root.winfo_children() if isinstance(child, CorridorTable))
        first, second = table.tree.get_children()
        table.tree.selection_set(first)
        table._open_selected()
        table.tree.selection_set(second)
        table._open_selected()
        assert opened == [(buffered_section, 1)]
        assert table.row_buttons[second].instate(['disabled'])
        assert table.winfo_width() > 0 and table.winfo_height() > 0
        for child in root.winfo_children():
            child.destroy()
        summary = SummaryView(root, result)
        summary.pack(fill='both', expand=True)
        summary.toggle_details()
        root.update()
        assert any('5 m padding' in str(child.cget('text')) for child in summary.details.winfo_children())
    finally:
        root.destroy()
        ctk.set_widget_scaling(1.0)
