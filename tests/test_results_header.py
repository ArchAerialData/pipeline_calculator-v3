"""Native repair notices share the results header without clipping or stale callbacks."""
from tkinter import ttk

import customtkinter as ctk
import pytest

from pipeline_calculator.gui.pages.results_page import show
from pipeline_calculator.gui.preferences import StateBreakdownPreference
from pipeline_calculator.gui.results_header import ResultsContextHeader
from test_repair_ui import source, workflow
from test_state_breakdown_ui import descendants, sample_results
from test_ui_lifecycle import settle


def render(root, flow, preference=None, *, geography=True):
    results = sample_results()
    if not geography:
        results.pop('geography')
    show(root, version='test', current_file='sample.kmz', current_results=results,
         on_export=lambda: None, on_reanalyze=lambda: None, on_new_file=lambda: None,
         on_exit=lambda: None, on_open_corridor=lambda *args: None,
         repair_workflow=flow, state_preference=preference)
    settle(root, .3)
    return next(w for w in root.winfo_children() if isinstance(w, ResultsContextHeader))


def assert_inside(widget, parent):
    assert widget.winfo_viewable()
    assert widget.winfo_rootx() >= parent.winfo_rootx()
    assert widget.winfo_rooty() >= parent.winfo_rooty()
    assert widget.winfo_rootx() + widget.winfo_width() <= parent.winfo_rootx() + parent.winfo_width() + 1
    assert widget.winfo_rooty() + widget.winfo_height() <= parent.winfo_rooty() + parent.winfo_height() + 1


@pytest.mark.native_gui
@pytest.mark.parametrize('scale', [1, 1.5, 2, 2.5])
def test_repair_notice_resizes_beside_scope_and_preserves_identity(monkeypatch, scale):
    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling', classmethod(lambda cls, window: scale))
    root = ctk.CTk()
    root.minsize(1, 1)
    root.geometry('1200x760')
    flow, calls = workflow(root)
    flow.source = source(verified=True, can_save=True, report={'status': 'verified'})
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        header = render(root, flow)
        notice = header.repair_notice
        for _ in range(2):
            for width in (1200, 620, 1200):
                root.geometry(f'{width}x760')
                settle(root, .2)
                assert_inside(header.selector, header)
                assert_inside(notice, header)
                if width == 1200:
                    assert notice.winfo_rootx() >= header.scope.winfo_rootx() + header.scope.winfo_width()
                    assert abs(notice.winfo_rooty() - header.scope.winfo_rooty()) <= 1
                    assert header.winfo_height() <= max(header.scope.winfo_height(), notice.winfo_height()) + 2
                else:
                    assert notice.winfo_rooty() >= header.scope.winfo_rooty() + header.scope.winfo_height(), (
                        scale, root.winfo_width(), header.winfo_width(), header._layout_key)
                for button in descendants(notice):
                    if isinstance(button, ctk.CTkButton):
                        assert_inside(button, notice)
                        assert button._text_label.winfo_reqwidth() <= button.winfo_width() - 8 * scale
                for label in descendants(header):
                    if isinstance(label, ctk.CTkLabel) and label.winfo_viewable():
                        assert label._label.winfo_reqwidth() <= label.winfo_width() + 2
        for state in ('Texas', 'Combined'):
            header.selector.set(state)
            header.selector.event_generate('<<ComboboxSelected>>')
            settle(root)
            assert header.repair_notice is notice and notice.winfo_viewable()
        assert not errors
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
def test_no_scope_or_notice_leaves_no_space_and_live_notices_clean_up(tmp_path):
    root = ctk.CTk()
    root.geometry('1000x700')
    flow, calls = workflow(root)
    preference = StateBreakdownPreference(root, path=tmp_path / 'preferences.json')
    original_traces = preference.notice.trace_info()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        header = render(root, flow, preference, geography=False)
        assert header.selector is None and not header.winfo_manager()
        preference.notice.set('This setting is active for this session, but could not be saved.')
        settle(root)
        assert header.winfo_viewable() and header.notices.winfo_viewable()
        assert header.notices.grid_info()['row'] == 0
        assert header.notices.grid_info()['column'] == 0
        preference.notice.set('')
        settle(root)
        assert not header.winfo_manager()
        header.destroy()
        assert preference.notice.trace_info() == original_traces
        # Restoring a notice with state scope places it alongside the selector.
        header = render(root, flow, preference)
        preference.notice.set('Could not save this setting. ' * 4)
        settle(root)
        assert header.notices.grid_info()['column'] == 1
        preference.notice.set('')
        settle(root)
        assert not header.notices.winfo_manager() and header.scope.winfo_viewable()
        header._queue_layout()
        header.destroy()
        preference.notice.set('Late warning after navigation')
        settle(root)
        assert preference.notice.trace_info() == original_traces
        assert not errors
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
def test_repair_notice_variants_without_state_selector():
    root = ctk.CTk()
    root.geometry('900x650')
    flow, calls = workflow(root)
    try:
        for state, message, rules in (
            ('completed', 'File type recognized', ['filename_format_mismatch_v1']),
            ('failed', 'Analysis could not finish', []),
            ('cancelled', 'Analysis cancelled', []),
        ):
            flow.source = source(verified=True, can_save=False,
                                 report={'status': 'verified', 'rules': rules})
            flow.analysis_state = state
            header = render(root, flow, geography=False)
            assert header.selector is None
            assert not any(isinstance(w, ttk.Combobox) for w in descendants(root))
            assert header.notices.winfo_x() == 0
            assert header.notices.winfo_width() == header.winfo_width()
            assert any(message in w.cget('text') for w in descendants(header)
                       if isinstance(w, ctk.CTkLabel))
            buttons = [w for w in descendants(header) if isinstance(w, ctk.CTkButton)]
            assert next(w for w in buttons if w.cget('text') == 'Save repaired copy…').cget('state') == 'disabled'
    finally:
        flow.close()
        root.destroy()
