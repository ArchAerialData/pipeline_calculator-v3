from __future__ import annotations

import itertools
import sys
import zipfile
from types import SimpleNamespace

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.parsers import kml_kmz


def pipes(analyzer, offsets, start=(-100, 40), bearing=0, shift=0):
    result = []
    for i, offset in enumerate(offsets):
        point = analyzer.geod.fwd(*start, bearing + 90, offset)[:2]
        if i:
            point = analyzer.geod.fwd(*point, bearing, shift)[:2]
        end = analyzer.geod.fwd(*point, bearing, 300)[:2]
        result.append({'name': str(i), 'coordinates': [point, end]})
    return result


def calculate(analyzer, pipelines):
    return analyzer.calculate_overlap_results(pipelines, analyzer.find_parallel_segments(pipelines))


@pytest.mark.parametrize('order', list(itertools.permutations(range(3))))
def test_three_pipeline_chain_uses_two_passes_regardless_of_input_order(order):
    analyzer = PipelineAnalyzer()
    pipelines = pipes(analyzer, [0, 10, 20])
    result = calculate(analyzer, [pipelines[i] for i in order])
    assert result['savings_meters'] == pytest.approx(300, abs=5)


@pytest.mark.parametrize('shift', [0.5, 1.25, 2.5])
def test_offset_samples_do_not_hide_parallel_overlap(shift):
    analyzer = PipelineAnalyzer(detection_range=3)
    result = calculate(analyzer, pipes(analyzer, [0, 2], shift=shift))
    assert len(result['bundled_sections']) == 1
    assert result['savings_meters'] == pytest.approx(300 - shift, abs=5)


def test_end_to_end_lines_are_not_bundled():
    analyzer = PipelineAnalyzer(min_parallel_length=10)
    result = calculate(analyzer, pipes(analyzer, [0, 0], shift=301))
    assert result['bundled_sections'] == []
    assert result['savings_meters'] == 0


@pytest.mark.parametrize('start,bearing', [((179.999, 60), 90), ((-179.999, 60), 270), ((0, 89.99), 90)])
def test_corridor_center_and_vertices_stay_near_source(start, bearing):
    analyzer = PipelineAnalyzer()
    result = calculate(analyzer, pipes(analyzer, [0, 2], start, bearing))
    section = result['bundled_sections'][0]
    assert analyzer.geod.inv(*start, section['center_lon'], section['center_lat'])[2] < 350
    for point in section['corridor_polygon']:
        assert -180 <= point[0] <= 180 and -90 <= point[1] <= 90
        assert analyzer.geod.inv(*start, *point)[2] < 400


@pytest.mark.parametrize('linked', [None, 'not xml'])
def test_missing_or_malformed_linked_kml_marks_totals_incomplete(tmp_path, linked):
    path = tmp_path / 'root.kml'
    path.write_text('<kml><Placemark><LineString><coordinates>-100,40 -100,40.001</coordinates>'
                    '</LineString></Placemark><NetworkLink><Link><href>other.kml</href></Link>'
                    '</NetworkLink></kml>')
    if linked is not None:
        (tmp_path / 'other.kml').write_text(linked)
    result = PipelineAnalyzer().analyze_complete(path)
    assert result['total_meters'] > 0
    assert result['analysis_complete'] is False


def test_empty_input_is_not_a_successful_zero_total(tmp_path):
    path = tmp_path / 'empty.kml'
    path.write_text('<kml/>')
    assert PipelineAnalyzer().analyze_complete(path)['analysis_complete'] is False


def test_compressed_kmz_limit_applies_to_decompressed_size(tmp_path, monkeypatch):
    monkeypatch.setattr(kml_kmz, 'MAX_KML_BYTES', 100)
    path = tmp_path / 'large.kmz'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('doc.kml', '<kml>' + ' ' * 5000 + '</kml>')
    with pytest.raises(ValueError, match='input limit'):
        PipelineAnalyzer().analyze_complete(path)


def test_linked_document_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(kml_kmz, 'MAX_KML_DOCUMENTS', 1)
    (tmp_path / 'root.kml').write_text('<kml><NetworkLink><Link><href>other.kml</href></Link></NetworkLink></kml>')
    (tmp_path / 'other.kml').write_text('<kml/>')
    with pytest.raises(ValueError, match='document-count'):
        PipelineAnalyzer().analyze_complete(tmp_path / 'root.kml')


def test_analysis_size_and_neighbor_budgets(monkeypatch):
    from pipeline_calculator.core import overlap, segmentation
    analyzer = PipelineAnalyzer()
    monkeypatch.setattr(segmentation, 'MAX_ANALYSIS_SEGMENTS', 5)
    with pytest.raises(ValueError, match='segment limit'):
        calculate(analyzer, pipes(analyzer, [0, 2]))
    monkeypatch.setattr(segmentation, 'MAX_ANALYSIS_SEGMENTS', 1000)
    monkeypatch.setattr(overlap, 'MAX_CANDIDATE_CHECKS', 1)
    with pytest.raises(ValueError, match='Neighbor-search limit'):
        calculate(analyzer, pipes(analyzer, [0, 2]))


def test_geodesic_failure_cannot_substitute_invented_midpoint():
    from pipeline_calculator.core.segmentation import segment_pipeline
    analyzer = PipelineAnalyzer()
    coords = pipes(analyzer, [0])[0]['coordinates']
    class FailingGeod:
        calls = 0
        def inv(self, *args):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError('midpoint calculation failed')
            return analyzer.geod.inv(*args)
        def fwd(self, *args):
            return analyzer.geod.fwd(*args)
    with pytest.raises(ValueError, match='midpoint calculation failed'):
        segment_pipeline(FailingGeod(), coords, 5)


def test_busy_gui_ignores_reentrant_analysis_without_changing_file():
    from pipeline_calculator.gui.main_window import PipelineCalculatorGUI
    gui = PipelineCalculatorGUI.__new__(PipelineCalculatorGUI)
    gui._processing = True
    gui.state = SimpleNamespace(current_file='original.kmz')
    gui.process_file('different.kmz')
    assert gui.state.current_file == 'original.kmz'


def test_gui_analysis_start_failure_resets_busy_state(monkeypatch):
    from pipeline_calculator.gui import main_window
    from pipeline_calculator.gui.state import AppState
    gui = main_window.PipelineCalculatorGUI.__new__(main_window.PipelineCalculatorGUI)
    gui._processing = False
    gui.state = AppState(current_results={'old': True})
    def fail():
        raise ValueError('bad parameters')
    gui._get_parameters = fail
    gui.show_file_selection = lambda: None
    monkeypatch.setattr(main_window.messagebox, 'showerror', lambda *a, **k: None)
    gui.process_file('new.kmz')
    assert gui._processing is False
    assert gui.state.current_results is None


def test_duplicate_normalized_kml_entries_are_rejected(tmp_path):
    path = tmp_path / 'duplicate.kmz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('doc.kml', '<kml/>')
        archive.writestr('./doc.kml', '<kml/>')
    with pytest.raises(ValueError, match='duplicate KML'):
        PipelineAnalyzer().analyze_complete(path)


@pytest.mark.parametrize('offsets,expected', [([0, 4, 8, 12], 900), ([0, 2, 100, 102], 600)])
def test_multiple_mutually_compatible_groups(offsets, expected):
    analyzer = PipelineAnalyzer()
    result = calculate(analyzer, pipes(analyzer, offsets))
    assert result['savings_meters'] == pytest.approx(expected, abs=10)


@pytest.mark.native_gui
@pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows widget smoke test')
def test_overlap_pagination_native_widgets():
    import customtkinter as ctk
    from tkinter import ttk
    from pipeline_calculator.gui.tabs.overlap_tab import create
    root = ctk.CTk()
    root.withdraw()
    try:
        results = {'overlap_analysis': {'bundled_sections': [
            {'pipeline_1': f'A{i}', 'pipeline_2': 'B', 'bundled_length_miles': 1,
             'average_separation': 2} for i in range(45)
        ]}}
        create(root, results, on_open_corridor=lambda *args: None)
        def widgets(node):
            for child in node.winfo_children():
                yield child
                yield from widgets(child)
        root.update_idletasks()
        tree = next(w for w in widgets(root) if isinstance(w, ttk.Treeview))
        buttons = {w.cget('text'): w for w in widgets(root) if isinstance(w, ctk.CTkButton)}
        assert len(tree.get_children()) == 20
        buttons['Next'].invoke()
        root.update_idletasks()
        assert tree.item(tree.get_children()[0])['values'][0] == 'A20 + B'
        buttons['Next'].invoke()
        root.update_idletasks()
        assert len(tree.get_children()) == 5
        assert tree.item(tree.get_children()[0])['values'][0] == 'A40 + B'
        assert buttons['Next'].cget('state') == 'disabled'
        buttons['Previous'].invoke()
        root.update_idletasks()
        assert tree.item(tree.get_children()[0])['values'][0] == 'A20 + B'
    finally:
        root.destroy()


def test_exported_source_text_cannot_become_an_excel_formula(tmp_path):
    from openpyxl import load_workbook
    from pipeline_calculator.export.xlsx import build_analysis_workbook
    workbook = build_analysis_workbook({
        'pipelines': [{'Name': '=1+1', 'OBJECTID': '=2+2', 'pipelinelength': 1.0}],
        'diagnostics': [{'level': 'warning', 'message': '=3+3'}],
    })
    path = tmp_path / 'literal-values.xlsx'
    workbook.save(path)
    saved = load_workbook(path)
    assert saved['Pipeline Length Analysis']['A2'].data_type == 's'
    assert saved['Pipeline Length Analysis']['B2'].value == '=1+1'
    assert saved['Pipeline Length Analysis']['B2'].data_type == 's'
    assert saved['Diagnostics']['C2'].data_type == 's'
    assert saved['Pipeline Length Analysis']['D2'].data_type == 'f'
