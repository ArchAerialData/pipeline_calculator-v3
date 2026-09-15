"""Native regressions for ownership, remapping and recoverable result views."""
import gc
import time
import weakref
import json
import os
import platform
from pathlib import Path

import customtkinter as ctk
import pytest

from pipeline_calculator.gui.layout import WrappedLabel, ResultPages
from pipeline_calculator.gui.scrolling import AutoScrollFrame


def settle(root, seconds=.08):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        root.update()
        time.sleep(.002)


def report(root, name, data):
    data.update(python=platform.python_version(), platform=platform.system(),
                tk=root.tk.call('info', 'patchlevel'), customtkinter=ctk.__version__,
                strict_performance_gate=os.environ.get('PIPELINE_UI_PERFORMANCE_GATE') == '1')
    print(json.dumps({name: data}))
    if os.environ.get('PIPELINE_UI_REPORT_DIR'):
        directory = Path(os.environ['PIPELINE_UI_REPORT_DIR'])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{name}.json').write_text(json.dumps(data, indent=2), encoding='utf-8')


def ui_budget(actual, target):
    """Strict UX budgets run explicitly on reference hardware, outside builds.

    Ordinary CI still catches long stalls; desktop scheduling/DWM contention
    must not turn a functional regression run into a hardware speed test.
    """
    limit = target if os.environ.get('PIPELINE_UI_PERFORMANCE_GATE') == '1' else 2
    assert actual <= limit, {'elapsed_seconds': actual, 'limit_seconds': limit}


def show_results(root):
    from test_state_breakdown_ui import sample_results
    from pipeline_calculator.gui.pages.results_page import show
    show(root, version='test', current_file='sample.kmz', current_results=sample_results(),
         on_export=lambda: None, on_reanalyze=lambda: None, on_new_file=lambda: None,
         on_exit=lambda: None, on_open_corridor=lambda *args: None)


@pytest.mark.native_gui
def test_retired_scroll_frames_release_global_bindings():
    root = ctk.CTk()
    root.geometry('640x480')
    events = ('<MouseWheel>', '<KeyPress-Shift_L>', '<KeyRelease-Shift_L>')
    sentinel = []
    root.bind_all('<MouseWheel>', lambda event: sentinel.append(event.delta), add='+')
    baseline = {event: root.bind_all(event) for event in events}
    refs = []
    try:
        for _ in range(20):
            host = ctk.CTkFrame(root)
            host.pack(fill='both', expand=True)
            view = AutoScrollFrame(host)
            view.pack(fill='both', expand=True)
            refs.append(weakref.ref(view))
            root.update()
            host.destroy()
        del host, view
        settle(root)
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert {event: root.bind_all(event) for event in events} == baseline
        root.event_generate('<MouseWheel>', delta=120)
        assert sentinel == [120], 'Cleanup must preserve unrelated handlers'
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_retired_labels_detach_from_live_parent():
    root = ctk.CTk()
    host = ctk.CTkFrame(root)
    host.pack()
    target = host._canvas
    baseline = target.bind('<Configure>')
    refs = []
    try:
        for _ in range(20):
            label = WrappedLabel(host, text='test')
            refs.append(weakref.ref(label))
            label.destroy()
        del label
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert target.bind('<Configure>') == baseline
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_same_tab_does_not_unmap_content():
    root = ctk.CTk()
    pages = ResultPages(root)
    pages.pack(fill='both', expand=True)
    page = pages.add('Summary')
    pages.add('Other')
    events = []
    # CTkFrame.bind redirects to its canvas; listen to the actual page mapping.
    import tkinter as tk
    tk.Misc.bind(page, '<Unmap>', lambda event: events.append(event), add='+')
    try:
        settle(root)
        for _ in range(20):
            pages.set('Summary')
            root.update()
        assert not events
        assert page.winfo_viewable()
        control = tk.Button(page, text='Focusable')
        control.pack()
        settle(root)
        control.focus_force()
        pages.set('Other')
        settle(root)
        assert root.focus_get() is not control
        assert root.focus_get().winfo_viewable()
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_summary_repeated_returns_remain_visible_and_fast():
    from test_state_breakdown_ui import descendants
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView
    root = ctk.CTk()
    root.geometry('1280x800')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    try:
        show_results(root)
        settle(root, .4)
        pages = next(w for w in root.winfo_children() if isinstance(w, ResultPages))
        view = next(w for w in descendants(root) if isinstance(w, SummaryView))
        original = view.original.value.cget('text')
        times = []
        reflow_times = []
        beats = []
        def heartbeat():
            beats.append(time.perf_counter())
            root.after(10, heartbeat)
        heartbeat()
        for index in range(100):
            if index % 10 == 0:
                view.toggle.invoke()
            pages.set('Pipelines')
            root.update()
            if index % 20 == 0:
                root.geometry('640x480' if index % 40 == 0 else '1280x800')
                settle(root)
            started = time.perf_counter()
            pages.set('Summary')
            root.update()
            (reflow_times if index % 20 == 0 else times).append(time.perf_counter() - started)
            assert view.winfo_viewable() and view.inner.winfo_viewable()
            canvas = view._parent_canvas
            assert view.inner.winfo_rooty() < canvas.winfo_rooty() + canvas.winfo_height()
            assert view.inner.winfo_rooty() + view.inner.winfo_height() > canvas.winfo_rooty()
            assert view.original.value.cget('text') == original
            assert next(w for w in descendants(root) if isinstance(w, SummaryView)) is view
        settle(root, .5)
        assert view._arrange_id is None and view._refresh_id is None
        assert not errors, errors
        p95 = sorted(times)[int(len(times) * .95)]
        heartbeat_gap = max(b - a for a, b in zip(beats, beats[1:]))
        report(root, 'summary_returns', {'cycles': 100, 'warm_cycles': len(times), 'p95_ms': p95 * 1000,
                                        'max_ms': max(times) * 1000, 'callback_errors': errors,
                                        'resize_reflow_max_ms': max(reflow_times) * 1000,
                                        'heartbeat_max_gap_ms': heartbeat_gap * 1000,
                                        'measurement': 'mapped content after event dispatch; not physical screen paint'})
        ui_budget(p95, .1)
        ui_budget(max(times), .25)
        ui_budget(max(reflow_times), .5)
        ui_budget(heartbeat_gap, .25)
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_scope_replacement_releases_views_and_recovers_from_failure(monkeypatch):
    from tkinter import ttk
    from test_state_breakdown_ui import descendants
    from pipeline_calculator.gui.pages import results_page
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView
    root = ctk.CTk()
    root.geometry('1000x720')
    errors = []
    root.report_callback_exception = lambda *error: errors.append(str(error))
    refs = []
    try:
        show_results(root)
        settle(root)
        selector = next(w for w in descendants(root) if isinstance(w, ttk.Combobox))
        def choose(name):
            selector.set(name)
            selector.event_generate('<<ComboboxSelected>>')
            root.update()
        before = next(w for w in descendants(root) if isinstance(w, SummaryView))
        choose('Combined')
        assert next(w for w in descendants(root) if isinstance(w, SummaryView)) is before
        constructor = results_page.create_summary_tab
        def fail(*args, **kwargs):
            raise ValueError('Injected render failure')
        monkeypatch.setattr(results_page, 'create_summary_tab', fail)
        choose('Texas')
        assert selector.get() == 'Combined'
        assert before.winfo_viewable()
        assert any(isinstance(w, ctk.CTkLabel) and 'Could not display Texas' in w.cget('text')
                   and w.winfo_viewable() for w in descendants(root))
        monkeypatch.setattr(results_page, 'create_summary_tab', constructor)
        del before
        times = []
        beats = []
        def heartbeat():
            beats.append(time.perf_counter())
            root.after(10, heartbeat)
        heartbeat()
        for index in range(100):
            refs.append(weakref.ref(next(w for w in descendants(root) if isinstance(w, SummaryView))))
            started = time.perf_counter()
            choose('Texas' if index % 2 == 0 else 'Combined')
            times.append(time.perf_counter() - started)
        settle(root, .5)
        gc.collect()
        assert not any(ref() is not None for ref in refs)
        bindings = root.bind_all('<MouseWheel>')
        assert bindings.count('_mouse_wheel_all') == 1
        assert not errors, errors
        # Sizes/styles plateau across view replacement rather than using widget IDs.
        assert len(root._pipeline_table_styles) <= 4
        gap = max(b-a for a, b in zip(beats, beats[1:]))
        report(root, 'scope_replacement', {'cycles': 100, 'retained_retired_views': 0,
                                         'max_ms': max(times) * 1000, 'style_variants': len(root._pipeline_table_styles),
                                         'callback_errors': errors, 'heartbeat_max_gap_ms': gap*1000})
        ui_budget(gap, .25)
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_disclosure_pointer_keyboard_focus_and_disabled_states():
    from pipeline_calculator.gui.disclosure import DisclosureButton
    root = ctk.CTk()
    activations = []
    button = DisclosureButton(root, text='Additional details', command=lambda: activations.append(1),
                              bg='#272D35', fg='#F1F4F8', takefocus=True,
                              highlightthickness=1, highlightcolor='#9CC8EB')
    button.pack()
    try:
        settle(root)
        button.focus_force()
        button.event_generate('<KeyPress-Return>')
        button.event_generate('<KeyPress-Return>')
        button.event_generate('<KeyRelease-Return>')
        assert len(activations) == 1
        button.event_generate('<KeyPress-space>')
        button.event_generate('<KeyRelease-space>')
        assert len(activations) == 2
        button.event_generate('<ButtonPress-1>', x=2, y=2)
        button.event_generate('<ButtonRelease-1>', x=-2, y=-2)
        assert len(activations) == 2
        button.event_generate('<ButtonPress-1>', x=2, y=2)
        button.event_generate('<ButtonRelease-1>', x=2, y=2)
        assert len(activations) == 3
        button.configure(state='disabled')
        button.event_generate('<Enter>')
        button.invoke()
        assert button.cget('background') == '#272D35'
        assert len(activations) == 3
    finally:
        root.destroy()
