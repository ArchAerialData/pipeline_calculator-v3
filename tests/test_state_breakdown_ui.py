"""User preferences, immutable job options, and scoped desktop presentation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import time

import pytest

from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.core.constants import SURVEY_MILE_METERS
from pipeline_calculator.gui import preferences
from pipeline_calculator.gui.controllers import analysis_controller as controller
from pipeline_calculator.gui.state import AnalysisParameters


def test_preferences_roundtrip_and_invalid_files_default_off(tmp_path):
    path = tmp_path / 'configuration' / 'preferences.json'
    assert preferences.load_state_breakdown(path) is False
    preferences.save_state_breakdown(True, path)
    assert preferences.load_state_breakdown(path) is True
    assert json.loads(path.read_text()) == {'schema_version': 1, 'state_breakdown': True}
    preferences.save_state_breakdown(False, path)
    assert preferences.load_state_breakdown(path) is False
    for value in ('{', '[]', 'null', '{"schema_version":2,"state_breakdown":true}',
                  '{"schema_version":1,"state_breakdown":"true"}'):
        path.write_text(value)
        assert preferences.load_state_breakdown(path) is False


def test_failed_atomic_save_preserves_previous_setting(tmp_path, monkeypatch):
    path = tmp_path / 'preferences.json'
    preferences.save_state_breakdown(True, path)
    def fail(*args):
        raise PermissionError('read only')
    monkeypatch.setattr(preferences.os, 'replace', fail)
    with pytest.raises(PermissionError):
        preferences.save_state_breakdown(False, path)
    assert preferences.load_state_breakdown(path) is True
    assert list(tmp_path.iterdir()) == [path]


def test_preference_save_failure_keeps_current_session_enabled(tmp_path, monkeypatch):
    import tkinter as tk
    binding = preferences.StateBreakdownPreference(tk.Tcl(), path=tmp_path / 'preferences.json')
    def fail(*args):
        raise PermissionError('read only')
    monkeypatch.setattr(preferences, 'save_state_breakdown', fail)
    binding.commit(True)
    assert binding.snapshot().state_breakdown is True
    assert 'could not be saved' in binding.notice.get()


def test_job_snapshots_options_before_start(monkeypatch):
    received = []
    def analyze(path, params, *, context, options, **kwargs):
        received.append(options)
        return {'total_miles': 1}
    monkeypatch.setattr(controller, 'analyze_file', analyze)
    options = AnalysisOptions(state_breakdown=True)
    with pytest.raises(FrozenInstanceError):
        options.state_breakdown = False
    job = controller.AnalysisJob('source.kml', AnalysisParameters(), options=options)
    job.options = AnalysisOptions(state_breakdown=False)
    job.start()
    assert job.done.wait(5)
    assert job.error is None
    assert received == [options]


def test_controller_forwards_enabled_options_to_core(monkeypatch):
    received = []
    class Analyzer:
        def __init__(self, **kwargs):
            pass
        def analyze_complete(self, path, **kwargs):
            received.append(kwargs)
            return {}
    monkeypatch.setattr(controller, 'PipelineAnalyzer', Analyzer)
    options = AnalysisOptions(state_breakdown=True)
    controller.analyze_file('source.kml', AnalysisParameters(), options=options)
    assert received == [{'context': None, 'options': options}]


def sample_results():
    def state(name, code, interior, allocated, *, complete=True):
        total = interior + allocated
        return {'state_name': name, 'state_code': code, 'interior_meters': interior,
                'shared_allocation_meters': allocated, 'total_meters': total,
                'total_miles': total / SURVEY_MILE_METERS, 'analysis_complete': complete,
                'adjusted_total_meters': total if complete else None,
                'interior_savings_meters': 0 if complete else None,
                'pipelines': [{'Name': name + ' segment', 'Placemark_ID': name,
                               'source_id': 'source-1', 'Shape_Length': total,
                               'interior_meters': interior, 'shared_allocation_meters': allocated,
                               'pipelinelength': total / SURVEY_MILE_METERS}],
                'placemarks': [], 'overlap_analysis': None, 'diagnostics': []}
    texas = state('Texas', 'TX', 100, 0.01)
    oklahoma = state('Oklahoma', 'OK', 100, 0.01, complete=False)
    return {'analysis_complete': True, 'total_miles': 205.02 / SURVEY_MILE_METERS,
            'total_meters': 205.02, 'pipelines': [{'Name': 'Original', 'Shape_Length': 205.02}],
            'placemarks': [{'Name': 'Source point'}], 'overlap_analysis': None, 'diagnostics': [],
            'geography': {'schema_version': 1, 'status': 'incomplete', 'analysis_complete': False,
                          'states': [texas, oklahoma], 'diagnostics': [],
                          'boundary_source': {'name': 'Census TIGER/Line', 'vintage': '2025'},
                          'reconciliation': {'passed': True, 'source_meters': 205.02,
                                             'attributed_state_meters': 200.02,
                                             'shared_meters': .02,
                                             'outside_meters': 2, 'unresolved_meters': 3}}}


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def settle(root, seconds=.15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        time.sleep(.005)


@pytest.mark.native_gui
@pytest.mark.parametrize('implementation', ['modern', 'legacy'])
def test_both_desktop_entrypoints_render_state_results_and_export_whole_snapshot(tmp_path, monkeypatch, implementation):
    import customtkinter as ctk
    from tkinter import ttk
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView
    from pipeline_calculator.gui.layout import ResultPages
    from pipeline_calculator.gui.actions import export_actions
    from pipeline_calculator.gui import main_window
    import pipeline_calculator_v3 as legacy

    monkeypatch.setattr(preferences, 'preferences_path', lambda: tmp_path / 'preferences.json')
    gui_class = main_window.PipelineCalculatorGUI if implementation == 'modern' else legacy.PipelineCalculatorGUI
    app = gui_class()
    errors, exported = [], []
    app.root.report_callback_exception = lambda *args: errors.append(args)
    export = lambda data, path: exported.append(data)
    monkeypatch.setattr(main_window, 'export_with_dialog', export)
    monkeypatch.setattr(export_actions, 'export_with_dialog', export)
    snapshot = sample_results()
    original = deepcopy(snapshot)
    try:
        switches = [item for item in descendants(app.root) if isinstance(item, ctk.CTkSwitch)]
        assert len(switches) == 1 and switches[0].cget('text') == 'State breakdown'
        switches[0].toggle()
        assert preferences.load_state_breakdown(tmp_path / 'preferences.json')
        app.state_preference.set_busy(True)
        assert switches[0].cget('state') == 'disabled'
        app.state_preference.set_busy(False)
        if implementation == 'modern':
            app.state.current_results, app.state.current_file = snapshot, 'source.kml'
        else:
            app.current_results, app.current_file = snapshot, 'source.kml'
        app.show_results()
        settle(app.root)
        selector = next(item for item in descendants(app.root) if isinstance(item, ttk.Combobox))
        assert tuple(selector.cget('values')) == ('Combined', 'Oklahoma', 'Texas')
        summary = next(item for item in descendants(app.root) if isinstance(item, SummaryView))
        assert len(summary.state_table.get_children()) == 2
        visible = '\n'.join(item.cget('text') for item in descendants(summary.inner)
                            if isinstance(item, ctk.CTkLabel) and item.winfo_viewable())
        assert '<0.001 mi follows shared borders and is divided equally' in visible
        assert 'Outside coverage: 0.001 mi' in visible and 'Unresolved: 0.002 mi' in visible
        selector.set('Texas')
        selector.event_generate('<<ComboboxSelected>>')
        settle(app.root)
        summary = next(item for item in descendants(app.root) if isinstance(item, SummaryView))
        assert summary.original.title.cget('text') == 'Mileage assigned to Texas'
        shown = '\n'.join(item.cget('text') for item in descendants(summary) if isinstance(item, ctk.CTkLabel))
        assert 'Includes <0.001 mi of shared-border allocation.' in shown
        assert 'Shared-border overlap: Not calculated.' in shown
        pages = next(item for item in app.root.winfo_children() if isinstance(item, ResultPages))
        pages.set('Placemarks')
        assert any('Combined view only' in item.cget('text') for item in descendants(pages.pages['Placemarks'])
                   if isinstance(item, ctk.CTkLabel))
        app.export_results()
        assert exported == [snapshot] and exported[0] is snapshot
        selector.set('Oklahoma')
        selector.event_generate('<<ComboboxSelected>>')
        settle(app.root)
        summary = next(item for item in descendants(app.root) if isinstance(item, SummaryView))
        assert summary.adjusted.value.cget('text') == 'Unavailable'
        assert snapshot == original
        def fail_save(*args):
            raise PermissionError('read only')
        monkeypatch.setattr(preferences, 'save_state_breakdown', fail_save)
        app.state_preference.commit(True)
        settle(app.root)
        assert any('could not be saved' in item.cget('text') or
                   (item.cget('textvariable') and app.state_preference.notice.get() == 'This setting is active for this session, but could not be saved.')
                   for item in descendants(app.root) if isinstance(item, ctk.CTkLabel) and item.winfo_viewable())
        assert not errors
    finally:
        app.close()


@pytest.mark.native_gui
def test_parameter_toggle_is_draft_until_apply(tmp_path):
    import customtkinter as ctk
    from tkinter import StringVar
    from pipeline_calculator.gui.dialogs.params_dialog import ParamsDialog
    root = ctk.CTk()
    root.geometry('800x650')
    path = tmp_path / 'preferences.json'
    binding = preferences.StateBreakdownPreference(root, path=path)
    applied = []
    variables = {name: StringVar(root, value='10') for name in
                 ('detection_range_var', 'segment_length_var', 'min_parallel_var', 'angular_tolerance_var')}
    try:
        dialog = ParamsDialog(root, **variables, state_preference=binding,
                              on_apply=lambda: applied.append(binding.snapshot()))
        dialog.show()
        dialog._state_draft.set(True)
        dialog._cancel()
        assert binding.snapshot().state_breakdown is False
        assert not path.exists() and not applied
        dialog = ParamsDialog(root, **variables, state_preference=binding,
                              on_apply=lambda: applied.append(binding.snapshot()))
        dialog.show()
        dialog._state_draft.set(True)
        dialog._apply()
        assert applied == [AnalysisOptions(state_breakdown=True)]
        assert preferences.load_state_breakdown(path) is True
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_state_results_small_and_high_dpi_windows(monkeypatch):
    import customtkinter as ctk
    from tkinter import ttk
    from pipeline_calculator.gui.pages.results_page import show
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView
    from pipeline_calculator.gui.layout import ResultPages
    ctk.ScalingTracker.get_window_dpi_scaling = classmethod(lambda cls, window: 1)
    root = ctk.CTk()
    root.minsize(1, 1)
    root.maxsize(5000, 3000)
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        show(root, version='test', current_file='state-example.kml', current_results=sample_results(),
             on_export=lambda: None, on_reanalyze=lambda: None, on_new_file=lambda: None,
             on_exit=lambda: None, on_open_corridor=lambda *args: None)
        selector = next(item for item in descendants(root) if isinstance(item, ttk.Combobox))
        for width, height, scale in [(1280, 800, 2), (390, 844, 1), (640, 480, 1)]:
            ctk.set_widget_scaling(scale)
            ctk.set_window_scaling(scale)
            settle(root, 1.1)
            root.geometry(f'{width}x{height}+-8000+0')
            settle(root)
            for scope in ('Combined', 'Texas'):
                selector.set(scope)
                selector.event_generate('<<ComboboxSelected>>')
                settle(root)
                summary = next(item for item in descendants(root) if isinstance(item, SummaryView))
                assert selector.winfo_viewable()
                assert int(root.tk.splitlist(selector.cget('font'))[-1]) == -round(14 * scale)
                assert summary.winfo_height() > 100 * scale
                for card in (summary.original, summary.adjusted):
                    assert card.winfo_width() > 200 * scale
                    for label in descendants(card):
                        if isinstance(label, ctk.CTkLabel):
                            assert label._label.winfo_reqwidth() <= label.winfo_width() + 2
                if scope == 'Combined':
                    assert summary.state_table.cget('xscrollcommand')
                    # Clicking a state table row selects that exact state, with no analysis call.
                    row = next(item for item, name in summary.state_rows.items() if name == 'Texas')
                    summary.state_table.selection_set(row)
                    summary.state_table.focus_force()
                    settle(root)
                    summary.state_table.event_generate('<Return>')
                    settle(root)
                    assert selector.get() == 'Texas'
        assert not errors, errors
    finally:
        root.destroy()
        ctk.set_widget_scaling(1)
        ctk.set_window_scaling(1)
