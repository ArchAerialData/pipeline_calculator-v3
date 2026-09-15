"""Shared scroll, table and file-work behavior outside the Summary page."""
import threading
import time
from types import SimpleNamespace

import customtkinter as ctk
import pytest

from pipeline_calculator.gui.scrolling import AutoScrollFrame
from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.table_loading import load_rows
from test_ui_lifecycle import settle, report, ui_budget


@pytest.mark.native_gui
def test_large_table_pauses_when_hidden_and_cancels_on_destroy():
    root = ctk.CTk()
    root.geometry('800x600')
    host = ctk.CTkFrame(root)
    host.pack(fill='both', expand=True)
    tree = create_table(host, ('Name',), (250,))
    def rows():
        for index in range(20000):
            tree.insert('', 'end', values=(str(index),))
            yield
    try:
        load_rows(tree, rows())
        assert 0 < tree.row_loader.count <= 200
        host.pack_forget()
        settle(root)
        paused_count = tree.row_loader.count
        settle(root)
        assert tree.row_loader.count == paused_count
        beats = []
        def beat():
            beats.append(time.perf_counter())
            root.after(10, beat)
        beat()
        host.pack(fill='both', expand=True)
        deadline = time.monotonic() + 8
        while tree.row_loader.rows is not None and time.monotonic() < deadline:
            settle(root, .01)
        assert tree.row_loader.rows is None
        assert len(tree.get_children()) == 20000
        assert len(beats) >= 2
        gap = max(b-a for a, b in zip(beats, beats[1:]))
        ui_budget(gap, .25)
        report(root, 'large_table', {'rows': 20000, 'heartbeat_max_gap_ms': gap*1000})
        load_rows(tree, rows())
        loader = tree.row_loader
        host.destroy()
        settle(root)
        assert loader.pending is None and loader.rows is None
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_table_load_failure_is_visible():
    root = ctk.CTk()
    tree = create_table(root, ('Name',), (250,))
    def rows():
        tree.insert('', 'end', values=('Loaded row',))
        yield
        raise ValueError('Injected row failure')
    try:
        load_rows(tree, rows())
        settle(root)
        assert tree.row_loader.rows is None
        assert tree.row_loader.status.winfo_viewable()
        assert 'Could not load all rows' in tree.row_loader.status.cget('text')
        assert len(tree.get_children()) == 1
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_nested_table_wheel_does_not_scroll_summary():
    root = ctk.CTk()
    root.geometry('700x500')
    view = AutoScrollFrame(root)
    view.pack(fill='both', expand=True)
    tree = create_table(view, ('Name',), (250,))
    for index in range(100):
        tree.insert('', 'end', values=(index,))
    ctk.CTkFrame(view, height=1000).pack(fill='x')
    try:
        settle(root)
        view._parent_canvas.yview_moveto(.2)
        before = view._parent_canvas.yview()
        # Exercise the native class binding plus all-bindings dispatch.
        tree.event_generate('<MouseWheel>', delta=-120)
        settle(root)
        assert view._parent_canvas.yview() == before
        assert tree.yview()[0] > 0
        view._mouse_wheel_all(SimpleNamespace(widget='.native-popup', delta=-1, state=0))
        view._mouse_wheel_all(SimpleNamespace(widget=view, delta=-120, state=1))
        assert view._shift_pressed is True
        view._mouse_wheel_all(SimpleNamespace(widget=view, delta=-120, state=0))
        assert view._shift_pressed is False
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_background_file_action_keeps_event_loop_alive_and_surfaces_errors():
    from pipeline_calculator.gui.background_action import run_background_action
    root = ctk.CTk()
    root.geometry('600x400')
    main_thread = threading.get_ident()
    beats = []
    def beat():
        beats.append(time.perf_counter())
        root.after(10, beat)
    beat()
    def work():
        assert threading.get_ident() != main_thread
        time.sleep(.25)
        return 'exported'
    try:
        result, error = run_background_action(root, 'Export', 'Writing…', work)
        assert result == 'exported' and error is None
        assert len(beats) >= 5
        gap = max(b-a for a, b in zip(beats, beats[1:]))
        ui_budget(gap, .25)
        def fail():
            raise OSError('Injected write failure')
        result, error = run_background_action(root, 'Export', 'Writing…', fail)
        assert result is None and isinstance(error, OSError)
        assert root.grab_current() is None
        report(root, 'background_export', {'heartbeat_max_gap_ms': gap*1000, 'worker_thread_verified': True})
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_style_sizes_are_shared_without_cross_window_mutation():
    from tkinter import ttk
    from pipeline_calculator.gui.styles import table_styles, scope_style, corridor_button_style
    root = ctk.CTk()
    other = ctk.CTkToplevel(root)
    try:
        small = table_styles(root, 1, 32)
        large = table_styles(other, 2, 40)
        assert table_styles(other, 1, 32) == small
        style = ttk.Style(root)
        assert style.lookup(small+'.Treeview', 'rowheight') == 32
        assert style.lookup(large+'.Treeview', 'rowheight') == 80
        combo = scope_style(root, 1)
        assert style.lookup(combo, 'fieldbackground', ('readonly',)) == '#242424'
        button = corridor_button_style(root, 1)
        assert style.lookup(button, 'background', ('disabled',)) == '#343D47'
        assert style.lookup(button, 'foreground', ('disabled',)) == '#B6C0CE'
    finally:
        root.destroy()
