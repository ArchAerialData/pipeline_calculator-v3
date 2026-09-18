"""State scope shares the existing navigation height and survives content swaps."""
from tkinter import StringVar

import customtkinter as ctk
import pytest

from pipeline_calculator.gui.layout import ResultPages
from pipeline_calculator.gui.pages import results_page
from pipeline_calculator.gui.results_header import ResultsContextHeader
from test_results_header import assert_inside
from test_state_breakdown_ui import sample_results
from test_ui_lifecycle import settle


TABS = ('Summary', 'Pipelines', 'Overlap Analysis', 'Placemarks', 'Diagnostics')


@pytest.mark.native_gui
@pytest.mark.parametrize('scale', [1, 1.5, 2.5])
def test_scope_shares_navigation_row_without_increasing_height(monkeypatch, scale):
    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling',
                        classmethod(lambda cls, window: scale))
    root = ctk.CTk()
    root.minsize(1, 1)
    root.maxsize(5000, 3000)
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        for width, height in ((1200, 760), (620, 480), (390, 480), (390, 230), (1200, 760)):
            root.geometry(f'{width}x{height}+16+16')
            settle(root)
            baseline = None
            for state_scope in (False, True):
                selection = StringVar(root, value='Combined')
                options = dict(scopes=('Combined', 'North Carolina', 'Texas'),
                               selection=selection, on_select_scope=lambda name: None) if state_scope else {}
                pages = ResultPages(root, **options)
                pages.pack(fill='both', expand=True, padx=10, pady=5)
                for name in TABS:
                    pages.add(name)
                pages.set('Diagnostics')
                settle(root, .25)
                factor = ctk.ScalingTracker.get_widget_scaling(pages)
                logical_height = root.winfo_height() / ctk.ScalingTracker.get_window_scaling(root)
                budget = 34 if logical_height < 280 else 50
                nav_height = pages.navigation.winfo_height()
                content_offset = pages.content.winfo_rooty() - pages.winfo_rooty()
                assert nav_height <= round(budget * factor) + 1
                assert content_offset <= round(budget * factor) + 1
                visible = pages.tabs if pages.tabs.winfo_viewable() else pages.selector
                assert_inside(visible, pages.navigation)
                assert pages.pages['Diagnostics'].winfo_viewable()
                if not state_scope:
                    baseline = (content_offset, pages.content.winfo_height())
                else:
                    assert abs(content_offset - baseline[0]) <= 1
                    assert abs(pages.content.winfo_height() - baseline[1]) <= 1
                    assert_inside(pages.scope, pages.navigation)
                    assert_inside(pages.scope_selector, pages.navigation)
                    # The scope remains at the left; tab/menu controls never
                    # overlap it or move beneath it into a second row.
                    assert pages.scope.winfo_rootx() - pages.navigation.winfo_rootx() <= round(20 * factor)
                    assert pages.scope.winfo_rootx() + pages.scope.winfo_width() <= visible.winfo_rootx() + 1
                    assert max(pages.scope_selector.winfo_rooty(), visible.winfo_rooty()) < min(
                        pages.scope_selector.winfo_rooty() + pages.scope_selector.winfo_height(),
                        visible.winfo_rooty() + visible.winfo_height())
                    if pages.scope_helper.winfo_viewable():
                        assert_inside(pages.scope_helper, pages.navigation)
                    if pages.navigation.winfo_width() / factor <= 400:
                        assert pages.selector.winfo_viewable() and not pages.tabs.winfo_viewable()
                pages.destroy()
                settle(root)
        assert not errors, errors
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_scope_and_navigation_persist_across_successful_and_failed_content_swaps(monkeypatch):
    root = ctk.CTk()
    root.geometry('1000x720')
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        results_page.show(root, version='test', current_file='sample.kmz', current_results=sample_results(),
                          on_export=lambda: None, on_reanalyze=lambda: None, on_new_file=lambda: None,
                          on_exit=lambda: None, on_open_corridor=lambda *args: None)
        settle(root)
        pages = next(w for w in root.winfo_children() if isinstance(w, ResultPages))
        header = next(w for w in root.winfo_children() if isinstance(w, ResultsContextHeader))
        selector, navigation = pages.scope_selector, pages.navigation
        assert not header.winfo_manager(), 'A notice-free context must consume no separate row'
        pages.set('Pipelines')
        content = pages.content
        constructor = results_page.create_summary_tab

        def fail(*args, **kwargs):
            raise ValueError('Injected render failure')

        monkeypatch.setattr(results_page, 'create_summary_tab', fail)
        selector.set('Texas')
        selector.event_generate('<<ComboboxSelected>>')
        settle(root)
        assert selector.get() == 'Combined'
        assert pages.content is content and pages.pages['Pipelines'].winfo_viewable()
        assert pages.navigation is navigation and pages.scope_selector is selector
        monkeypatch.setattr(results_page, 'create_summary_tab', constructor)
        for state in ('Texas', 'Combined'):
            selector.set(state)
            selector.event_generate('<<ComboboxSelected>>')
            settle(root)
            assert next(w for w in root.winfo_children() if isinstance(w, ResultPages)) is pages
            assert pages.navigation is navigation and pages.scope_selector is selector
            assert pages.content is not content
            assert pages.selector.get() == pages.tabs.get() == 'Pipelines'
            assert pages.pages['Pipelines'].winfo_viewable()
            assert selector.get() == state
            assert_inside(selector, navigation)
            content = pages.content
        assert not errors, errors
    finally:
        root.destroy()
