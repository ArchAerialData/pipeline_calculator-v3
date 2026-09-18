"""Calculation failures must never look like a successful zero-overlap run."""
import copy

import customtkinter as ctk
import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.gui.tabs.overlap_tab import create
from test_ui_lifecycle import settle


@pytest.mark.native_gui
def test_real_overlap_failure_and_success_render_distinct_empty_states(monkeypatch):
    pipelines = [
        {'id': 0, 'name': 'A', 'coordinates': [(-100, 30), (-100.01, 30)]},
        {'id': 1, 'name': 'B', 'coordinates': [(-101, 30), (-101.01, 30)]},
    ]
    analyzer = PipelineAnalyzer()

    def fail(*args, **kwargs):
        raise RuntimeError('injected overlap failure')

    with monkeypatch.context() as patch:
        patch.setattr(analyzer, 'find_parallel_segments', fail)
        failed = analyzer.analyze_features(copy.deepcopy(pipelines))
    assert failed['analysis_complete'] is False and failed['overlap_analysis'] is None
    valid_empty = analyzer.analyze_features(copy.deepcopy(pipelines))
    assert valid_empty['analysis_complete'] is True
    assert valid_empty['overlap_analysis']['bundled_sections'] == []
    single = analyzer.analyze_features(copy.deepcopy(pipelines[:1]))
    assert single['analysis_complete'] is True and single['overlap_analysis'] is None
    unrelated_error = dict(single, analysis_complete=False,
                           diagnostics=[{'level': 'error', 'code': 'invalid_coordinate'}])
    cases = [
        (failed, 'Overlap analysis is unavailable. See Diagnostics for details.'),
        (dict(failed, state_code='TX', adjusted_total_meters=None),
         'State overlap analysis is unavailable. See Diagnostics for details.'),
        (dict(failed, state_code='TX', adjusted_total_meters=None,
              diagnostics=[{'level': 'error', 'code': 'state_analysis_failed'}]),
         'State overlap analysis is unavailable. See Diagnostics for details.'),
        (valid_empty, 'No bundled sections found with current parameters'),
        (single, 'No bundled sections found with current parameters'),
        (unrelated_error, 'No bundled sections found with current parameters'),
        (dict(single, state_code='TX', adjusted_total_meters=single['total_meters']),
         'No bundled sections found with current parameters'),
    ]
    root = ctk.CTk()
    root.geometry('390x300')
    try:
        for result, expected in cases:
            create(root, result, on_open_corridor=lambda *args: None)
            settle(root, .1)
            labels = [child for child in root.winfo_children() if isinstance(child, ctk.CTkLabel)]
            assert [label.cget('text') for label in labels] == [expected]
            assert labels[0]._label.winfo_reqwidth() <= labels[0].winfo_width() + 2
            assert labels[0]._label.winfo_reqheight() <= labels[0].winfo_height() + 2
            labels[0].destroy()
        from pipeline_calculator.gui.pages.results_page import show
        from pipeline_calculator.gui.layout import ResultPages
        show(root, version='test', current_file='failed.kml', current_results=failed,
             on_export=lambda: None, on_reanalyze=lambda: None, on_new_file=lambda: None,
             on_exit=lambda: None, on_open_corridor=lambda *args: None)
        settle(root)
        pages = next(child for child in root.winfo_children() if isinstance(child, ResultPages))
        assert 'Overlap Analysis' in pages.pages
        pages.set('Overlap Analysis')
        settle(root)
        labels = [child for child in pages.pages['Overlap Analysis'].winfo_children()
                  if isinstance(child, ctk.CTkLabel)]
        assert labels[0].cget('text') == 'Overlap analysis is unavailable. See Diagnostics for details.'
        assert labels[0].winfo_viewable()
    finally:
        root.destroy()
