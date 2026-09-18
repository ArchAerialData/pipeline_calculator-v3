"""Native regressions for canvas windows that become unmapped offscreen."""
import customtkinter as ctk
import pytest

from pipeline_calculator.gui.layout import ResultPages
from pipeline_calculator.gui.scrolling import AutoScrollFrame
from pipeline_calculator.gui.tabs.summary_tab import SummaryView
from pipeline_calculator.smoke import _visibility_details
from test_state_breakdown_ui import descendants
from test_ui_lifecycle import settle, show_results


def assert_content_visible(view):
    canvas = view._parent_canvas
    assert canvas.winfo_viewable(), _visibility_details(view)
    assert view.winfo_viewable(), _visibility_details(view)
    assert view.winfo_rooty() < canvas.winfo_rooty() + canvas.winfo_height()
    assert view.winfo_rooty() + view.winfo_height() > canvas.winfo_rooty()


@pytest.mark.native_gui
def test_summary_recovers_after_collapse_tab_switch_and_hidden_resize():
    root = ctk.CTk()
    root.geometry('640x480')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    try:
        show_results(root)
        settle(root, .4)
        pages = next(w for w in root.winfo_children() if isinstance(w, ResultPages))
        view = next(w for w in descendants(root) if isinstance(w, SummaryView))
        original = view.original.value.cget('text')
        view.toggle.invoke()
        settle(root, .15)
        view._parent_canvas.yview_moveto(1)
        settle(root)
        assert view._parent_canvas.yview()[0] > 0

        # A user can change tabs before a disclosure's deferred layout finishes.
        # Its height changes while hidden, leaving the prior scroll region behind.
        view.toggle.invoke()
        pages.set('Pipelines')
        root.update()
        root.geometry('1280x800')
        settle(root, .15)
        assert not view._parent_canvas.winfo_viewable()
        pages.set('Summary')
        root.update()
        assert_content_visible(view)
        settle(root, .15)
        assert_content_visible(view)
        assert view.inner.winfo_viewable()
        assert view.original.value.cget('text') == original
        assert next(w for w in descendants(root) if isinstance(w, SummaryView)) is view
        assert view._arrange_id is None and view._refresh_id is None
        assert not errors, errors
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_visible_viewport_recovers_its_offscreen_content_on_refresh_and_remap():
    root = ctk.CTk()
    root.geometry('600x400')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    try:
        view = AutoScrollFrame(root)
        view.pack(fill='both', expand=True)
        for index in range(8):
            ctk.CTkLabel(view, text=f'Item {index}', height=50).pack(fill='x')
        settle(root, .2)
        canvas = view._parent_canvas
        for recovery in ('refresh', 'remap'):
            # Model a stale region after hidden content shrinks. Tk unmaps the
            # canvas window when it lies entirely outside this visible viewport.
            canvas.configure(scrollregion=(0, 0, canvas.winfo_width(), 10000))
            canvas.yview_moveto(1)
            settle(root)
            assert canvas.winfo_viewable()
            assert not view.winfo_viewable(), _visibility_details(view)
            if recovery == 'refresh':
                view._schedule_scrollbar()
                settle(root)
            else:
                view.pack_forget()
                root.update()
                view.pack(fill='both', expand=True)
                root.update()
            assert_content_visible(view)
            assert tuple(map(float, canvas.cget('scrollregion').split())) == canvas.bbox('all')
        settle(root)
        assert view._refresh_id is None and view._scroll_position_id is None
        assert not errors, errors
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_hidden_or_destroyed_viewport_cancels_pending_maintenance():
    root = ctk.CTk()
    root.geometry('600x400')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    try:
        view = AutoScrollFrame(root)
        view.pack(fill='both', expand=True)
        ctk.CTkLabel(view, text='Viewport content', height=700).pack(fill='x')
        settle(root, .2)
        view._painted_fraction = None
        view._queue_scroll_position(*view._parent_canvas.yview())
        view._schedule_scrollbar()
        pending = (view._scroll_position_id, view._refresh_id)
        assert all(pending)
        view.pack_forget()
        root.update()
        assert not view._parent_canvas.winfo_viewable()
        assert view._scroll_position_id is None and view._refresh_id is None
        assert not set(pending).intersection(root.tk.call('after', 'info'))
        view._queue_scroll_position(0, 1)
        view._schedule_scrollbar()
        assert view._scroll_position_id is None and view._refresh_id is None

        view.pack(fill='both', expand=True)
        settle(root)
        assert_content_visible(view)
        view._painted_fraction = None
        view._queue_scroll_position(*view._parent_canvas.yview())
        view._schedule_scrollbar()
        pending = (view._scroll_position_id, view._refresh_id)
        assert all(pending)
        view.destroy()
        settle(root)
        assert not set(pending).intersection(root.tk.call('after', 'info'))
        assert not errors, errors
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_retired_summary_does_not_reschedule_from_surviving_canvas():
    root = ctk.CTk()
    root.geometry('600x400')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    try:
        view = SummaryView(root, {'total_miles': 1, 'pipelines': [], 'analysis_complete': True})
        view.pack(fill='both', expand=True)
        settle(root, .2)
        canvas = view._parent_canvas
        view._queue_padding()
        padding_timer = view._padding_id
        parent_binding = view._viewport_parent_binding.token
        assert padding_timer
        view.destroy()
        # CTk 5.2 destroys the content frame separately from its canvas wrapper.
        # Real queued native canvas notifications must not revive maintenance.
        assert canvas.winfo_exists() and not view.winfo_exists()
        assert view._padding_id is None
        assert padding_timer not in root.tk.call('after', 'info')
        assert parent_binding not in str(root.bind('<Configure>'))
        canvas.event_generate('<Configure>', when='tail')
        root.geometry('600x250')
        settle(root, .2)
        assert view._refresh_id is None and view._scroll_position_id is None
        assert not errors, errors
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_refresh_retires_if_viewport_policy_destroys_content(monkeypatch):
    root = ctk.CTk()
    root.geometry('600x400')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    try:
        view = SummaryView(root, {'total_miles': 1, 'pipelines': [], 'analysis_complete': True})
        view.pack(fill='both', expand=True)
        settle(root, .2)
        def resize_during_teardown():
            # Force the reentrant idle-work boundary deterministically, instead
            # of depending on the native window manager's event ordering.
            root.after_idle(view.destroy)
            root.update_idletasks()
        monkeypatch.setattr(view, '_resize_viewport', resize_during_teardown)
        view._schedule_scrollbar()
        settle(root, .2)
        assert not view.winfo_exists()
        assert view._refresh_id is None and view._scroll_position_id is None
        assert not errors, errors
    finally:
        root.destroy()
