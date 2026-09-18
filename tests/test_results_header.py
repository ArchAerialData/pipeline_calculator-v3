"""Native repair notices share the results header without clipping or stale callbacks."""
from tkinter import Misc, ttk
import traceback

import customtkinter as ctk
import pytest

from pipeline_calculator.gui.pages.results_page import show
from pipeline_calculator.gui.preferences import StateBreakdownPreference
from pipeline_calculator.gui.results_header import ResultsContextHeader
from pipeline_calculator.gui.layout import ResultPages
from test_repair_ui import activate_for_keyboard, source, workflow
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
        observed_layouts = set()
        available_widths = []
        for _ in range(2):
            for width in (1200, 620, 1200):
                root.geometry(f'{width}x760')
                settle(root, .2)
                assert_inside(header.selector, header)
                if header._compact:
                    assert_inside(header.repair_button, header)
                    assert not notice.winfo_viewable() and not header.scope_helper.winfo_viewable()
                    assert next(w for w in root.winfo_children() if isinstance(w, ResultPages)).winfo_viewable()
                    observed_layouts.add('compact')
                    continue
                assert_inside(notice, header)
                # The window manager may clamp a 3000-pixel (1200 at 250%)
                # request to the macOS display. Judge the responsive contract
                # against the real available width, not the requested geometry.
                factor = ctk.ScalingTracker.get_widget_scaling(header)
                available = header.winfo_width() / factor
                scope_minimum = max(360, (header.view_label.winfo_reqwidth() +
                                         header.selector.winfo_reqwidth()) / factor + 16)
                wide_enough = available >= scope_minimum + 24 + 440
                available_widths.append((available, wide_enough))
                if wide_enough:
                    observed_layouts.add('beside')
                    assert notice.winfo_rootx() >= header.scope.winfo_rootx() + header.scope.winfo_width()
                    assert abs(notice.winfo_rooty() - header.scope.winfo_rooty()) <= 1
                    assert header.winfo_height() <= max(header.scope.winfo_height(), notice.winfo_height()) + 2
                else:
                    observed_layouts.add('stacked')
                    assert notice.winfo_rooty() >= header.scope.winfo_rooty() + header.scope.winfo_height(), (
                        scale, root.winfo_width(), header.winfo_width(), header._layout_key)
                for button in descendants(notice):
                    if isinstance(button, ctk.CTkButton):
                        assert_inside(button, notice)
                        assert button._text_label.winfo_reqwidth() <= button.winfo_width() - 8 * scale
                for label in descendants(header):
                    if isinstance(label, ctk.CTkLabel) and label.winfo_viewable():
                        assert label._label.winfo_reqwidth() <= label.winfo_width() + 2
        assert observed_layouts.intersection({'stacked', 'compact'})
        if any(wide_enough for _, wide_enough in available_widths):
            assert 'beside' in observed_layouts
        print(f'Header scale={scale}: widths/layouts={available_widths}, branches={sorted(observed_layouts)}')
        for state in ('Texas', 'Combined'):
            header.selector.set(state)
            header.selector.event_generate('<<ComboboxSelected>>')
            settle(root)
            assert header.repair_notice is notice
            assert (header.repair_button if header._compact else notice).winfo_viewable()
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


@pytest.mark.native_gui
@pytest.mark.parametrize('analysis_state, status', [('completed', 'Details'),
    ('failed', 'Analysis failed'), ('cancelled', 'Analysis cancelled')])
def test_compact_context_preserves_details_warnings_and_result_space(monkeypatch, tmp_path,
                                                                  analysis_state, status):
    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling', classmethod(lambda cls, window: 2.5))
    root = ctk.CTk()
    root.minsize(1, 1)
    root.maxsize(600, 300)
    root.geometry('1200x760+16+16')
    flow, _ = workflow(root)
    flow.source = source(verified=True, can_save=True, report={'status': 'verified'})
    flow.analysis_state = analysis_state
    preference = StateBreakdownPreference(root, path=tmp_path / 'preferences.json')
    warning = 'This setting is active for this session, but could not be saved.'
    preference.notice.set(warning)
    errors, messages = [], []
    root.report_callback_exception = lambda *args: errors.append(args)
    monkeypatch.setattr('pipeline_calculator.gui.results_header.messagebox.showwarning',
                        lambda title, text, **kwargs: messages.append((title, text)))
    try:
        header = render(root, flow, preference)
        notice = header.repair_notice
        pages = next(w for w in root.winfo_children() if isinstance(w, ResultPages))
        assert header._compact and not notice.winfo_viewable()
        assert status in header.repair_button.cget('text')
        for widget in (header.selector, header.repair_button, header.preference_button):
            assert_inside(widget, header)
        assert_inside(pages, root)
        assert pages.content.winfo_viewable()
        assert pages.content.winfo_height() > 1
        factor = ctk.ScalingTracker.get_widget_scaling(header)
        # At least 100 logical pixels when this native viewport permits it.
        if root.winfo_height() / factor >= 300 and root.winfo_width() / factor >= 590:
            assert pages.winfo_height() / factor >= 100
        activate_for_keyboard(root)
        Misc.focus_set(header.repair_button)
        settle(root)
        assert root.focus_get() is header.repair_button
        header.repair_button.event_generate('<Return>')
        settle(root)
        assert flow.panel is not None
        assert any(button.cget('text') == 'Save repaired copy…' for button in flow.panel.footer.buttons)
        close = next(button for button in flow.panel.footer.buttons if button.cget('text') == 'Close')
        Misc.focus_set(close)
        close.event_generate('<Return>')
        settle(root)
        assert flow.panel is None and root.focus_get() is header.repair_button
        Misc.focus_set(header.preference_button)
        header.preference_button.event_generate('<Return>')
        settle(root)
        assert messages == [('State breakdown setting', warning)]

        # A larger logical viewport (including a lower-DPI monitor) restores
        # these same notices, without relying on a 1900-pixel-tall CI display.
        Misc.focus_set(header.repair_button)
        ctk.set_widget_scaling(.4)
        ctk.set_window_scaling(.4)
        root.maxsize(1200, 900)
        root.geometry('900x700+16+16')
        settle(root, .4)
        assert not header._compact and header.repair_notice is notice
        assert_inside(notice, header)
        assert header.scope_helper.winfo_viewable()
        assert root.focus_get() is header._first_button(notice)
        root.geometry('600x300+16+16')
        settle(root, .3)
        assert header._compact and root.focus_get() is header.repair_button
        assert_inside(header.repair_button, header)
        assert_inside(header.preference_button, header)
        preference.notice.set('')
        settle(root)
        assert not header.preference_button.winfo_viewable()
        root_binding = header._root_binding.token
        header.destroy()
        preference.notice.set('Late save error after navigation')
        settle(root)
        assert root_binding not in str(root.bind('<Configure>'))
        assert not errors, errors
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
def test_preference_only_compact_notice_restores_focus_to_visible_action(tmp_path):
    root = ctk.CTk()
    root.minsize(1, 1)
    root.geometry('600x200+16+16')
    flow, _ = workflow(root)
    preference = StateBreakdownPreference(root, path=tmp_path / 'preferences.json')
    preference.notice.set('This setting is active for this session, but could not be saved.')
    try:
        header = render(root, flow, preference, geography=False)
        assert header._compact
        activate_for_keyboard(root)
        Misc.focus_set(header.preference_button)
        settle(root)
        root.geometry('600x600+16+16')
        settle(root, .3)
        assert not header._compact
        assert root.focus_get() is header._first_button(root)
        assert root.focus_get().winfo_viewable()
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
@pytest.mark.parametrize('rules,expected', [
    (['missing_xsi_schema_namespace_v1'], 'File repaired · Details'),
    (['filename_format_mismatch_v1'], 'File type recognized · Details'),
])
def test_deferred_header_keeps_displayed_repair_identity_after_source_retirement(monkeypatch, rules, expected):
    root = ctk.CTk()
    root.geometry('700x500')
    flow, _ = workflow(root)
    displayed_source = source(verified=True, can_save=True, report={'status': 'verified', 'rules': rules})
    flow.source = displayed_source
    flow.analysis_state = 'completed'
    errors, actions = [], []
    root.report_callback_exception = lambda *args: errors.append(''.join(traceback.format_exception(*args)))
    monkeypatch.setattr(flow, 'show_details', lambda: actions.append(('details', flow.source)))
    monkeypatch.setattr(flow, 'save_copy', lambda: actions.append(('save', flow.source)))
    try:
        header = render(root, flow)
        expanded_buttons = {w.cget('text'): w for w in descendants(header.repair_notice)
                            if isinstance(w, ctk.CTkButton)}
        controls = [header.repair_button, expanded_buttons['Details'], expanded_buttons['Save repaired copy…']]
        assert header.repair_button.cget('text') == expected
        for button in controls:
            button.invoke()
        assert actions == [('details', displayed_source), ('details', displayed_source), ('save', displayed_source)]
        actions.clear()

        # Importing another file retires the workflow source before the prior
        # results screen is destroyed. A pending resize must remain safe then.
        header._queue_layout()
        flow.retire_source()
        settle(root)
        assert not errors, errors
        assert header.repair_button.cget('text') == expected
        assert header.repair_button.cget('state') == 'disabled'
        for button in controls:
            button.invoke()
        assert actions == []

        # A later verified source must not be shown/saved by a stale notice.
        flow.source = source(verified=True, can_save=True, display_name='Replacement.kmz',
                             report={'status': 'verified', 'rules': ['another_rule']})
        flow.analysis_state = 'failed'
        header._queue_layout()
        settle(root)
        assert not errors, errors
        assert header.repair_button.cget('text') == expected
        for button in controls:
            button.invoke()
        assert actions == []
        header.destroy()
        settle(root)
        assert not errors, errors
    finally:
        flow.close()
        root.destroy()
