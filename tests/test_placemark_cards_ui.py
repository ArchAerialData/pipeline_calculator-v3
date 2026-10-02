"""Native card layout, responsive scaling, provenance, and view lifecycle."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import customtkinter as ctk
import pytest

from pipeline_calculator.gui.tabs.placemarks_tab import PlacemarksView
from test_ui_lifecycle import ui_budget


def settle(root, seconds=.2):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        root.update()
        time.sleep(.005)


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def records():
    from pipeline_calculator.parsers.source import prepare_source
    fixture = Path(__file__).parent / 'fixtures/geography/antero_midstream_data.kmz'
    return prepare_source(fixture).fresh_parse().placemarks


def check_labels(root, view):
    scale = ctk.ScalingTracker.get_widget_scaling(view)
    for card in [view.total_card, *view.folder_cards]:
        assert card.winfo_width() >= 180 * scale
        for label in descendants(card):
            if isinstance(label, ctk.CTkLabel):
                assert label._label.winfo_reqwidth() <= label.winfo_width() + 2, label.cget('text')
                assert label._label.winfo_reqheight() <= label.winfo_height() + 2, label.cget('text')
                assert label.winfo_rootx() >= card.winfo_rootx()
                assert label.winfo_rootx() + label.winfo_width() <= card.winfo_rootx() + card.winfo_width() + 2
    for card in view.folder_cards:
        assert card.winfo_rootx() + card.winfo_width() <= view._parent_canvas.winfo_rootx() + view._parent_canvas.winfo_width() + 2
    overflow = view.winfo_reqheight() > view._parent_canvas.winfo_height() + 1
    assert bool(view._scrollbar.winfo_manager()) == overflow
    assert root.winfo_exists()


def capture(root, name, receipt):
    destination = os.environ.get('PIPELINE_PLACEMARK_EVIDENCE_DIR')
    if not destination:
        return
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (name + '.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        from PIL import ImageGrab
        parent = ctypes.windll.user32.GetParent
        parent.argtypes, parent.restype = [wintypes.HWND], wintypes.HWND
        ImageGrab.grab(window=parent(root.winfo_id())).save(folder / (name + '.png'))


@pytest.mark.native_gui
@pytest.mark.parametrize('scale', [1, 1.25, 1.5, 2])
def test_folder_cards_fit_laptop_desktop_widescreen_and_narrow_windows(monkeypatch, scale):
    # Uses real Tk font/geometry measurements and CTk's window DPI path, without
    # changing display preferences. Portable to native macOS CI as well.
    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling', classmethod(lambda cls, window: scale))
    root = ctk.CTk()
    root.minsize(1, 1)
    root.maxsize(6000, 4000)
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    snapshot = {'placemarks': records()}
    before = deepcopy(snapshot)
    view = PlacemarksView(root, snapshot)
    view.pack(fill='both', expand=True, padx=8, pady=8)
    try:
        for name, physical_width, physical_height in [('laptop', 1366, 768), ('desktop', 1920, 1080),
                                                       ('widescreen', 2560, 1080), ('narrow', 640, 640)]:
            # Keep the narrow test at least 320 logical px even at 200% DPI.
            root.geometry(f'{round(physical_width / scale)}x{round(physical_height / scale)}')
            settle(root, .65)
            assert view.total_card.value.cget('text') == '7,275'
            assert len(view.folder_cards) == 6
            assert [int(card.value.cget('text').replace(',', '')) for card in view.folder_cards] == [4192, 1906, 420, 477, 250, 30]
            assert view.folder_grid.columns == max(1, min(4, int(view.folder_grid.winfo_width() / scale // 280)))
            check_labels(root, view)
            capture(root, f'{name}-{int(scale*100)}', {
                'platform': sys.platform, 'scale': scale, 'requested_physical': [physical_width, physical_height],
                'actual_physical': [root.winfo_width(), root.winfo_height()], 'columns': view.folder_grid.columns,
                'total': view.inventory['total'], 'groups': len(view.folder_cards), 'status': 'passed',
            })
            view._parent_canvas.yview_moveto(1)
            settle(root)
            last = view.folder_cards[-1]
            assert last.winfo_rooty() + last.winfo_height() <= view._parent_canvas.winfo_rooty() + view._parent_canvas.winfo_height() + 2
            view._parent_canvas.yview_moveto(0)
        assert snapshot == before
        assert not errors, errors
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_long_names_duplicate_folders_and_unrecorded_or_empty_inventories(monkeypatch):
    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling', classmethod(lambda cls, window: 1))
    root = ctk.CTk()
    root.geometry('640x600')
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    long_name = 'Long facility name ' * 25
    path = ['A distant parent folder ' * 12, long_name]
    points = [{'folder_name': long_name, 'folder_path': path, 'folder_id': index, 'source_kml': 'doc.kml'} for index in ('1', '2')]
    points.extend([{'folder_path': [], 'folder_id': '', 'source_kml': 'nested/second.kml'},
                   {'Name': 'Legacy', 'source_kml': 'nested/second.kml'}])
    try:
        for point_inventory, expected in [(points, '4'), ([{'Name': 'Legacy'}], '1'),
                                           ([], '0'), (None, 'Not recorded')]:
            view = PlacemarksView(root, {'placemarks': point_inventory})
            view.pack(fill='both', expand=True)
            settle(root, .6)
            assert view.total_card.value.cget('text') == expected
            check_labels(root, view)
            if point_inventory and len(point_inventory) == 4:
                assert len(view.folder_cards) == 4
                assert 'Same-name folder 1 of 2' in view.folder_cards[0].caption.cget('text')
                assert 'Same-name folder 2 of 2' in view.folder_cards[1].caption.cget('text')
                assert 'Source: doc.kml' in view.folder_cards[0].caption.cget('text')
                assert view.folder_cards[2].title.cget('text') == 'No subfolder'
                assert view.folder_cards[3].title.cget('text') == 'Folder not recorded'
                assert 'Same-name folder' not in view.folder_cards[2].caption.cget('text')
                assert 'Same-name folder' not in view.folder_cards[3].caption.cget('text')
            elif point_inventory:
                assert view.folder_grid.columns == 1
                assert len(view.folder_cards) == 1
                assert view.folder_cards[0].title.cget('text') == 'Folder not recorded'
                assert abs(view.folder_cards[0].winfo_width() - view.folder_grid.winfo_width() + 12) <= 2
            else:
                assert not view.folder_cards
            view.destroy()
            settle(root)
        assert not errors, errors
    finally:
        root.destroy()


@pytest.mark.native_gui(timeout=60)
def test_many_folders_pause_loading_when_hidden_and_release_callbacks(monkeypatch):
    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling', classmethod(lambda cls, window: 1))
    root = ctk.CTk()
    root.geometry('1100x700')
    errors, beats = [], []
    root.report_callback_exception = lambda *args: errors.append(args)
    points = [{'folder_path': [f'Folder {index}'], 'folder_id': str(index), 'source_kml': 'doc.kml'} for index in range(120)]
    host = ctk.CTkFrame(root)
    host.pack(fill='both', expand=True)
    view = PlacemarksView(host, {'placemarks': points})
    view.pack(fill='both', expand=True)
    def beat():
        beats.append(time.perf_counter())
        root.after(10, beat)
    try:
        assert len(view.folder_cards) <= 8
        # Hide using a real UI timer: update() may keep dispatching ready events
        # for several batches, so a wall-clock check after it returns is too late.
        root.after(40, host.pack_forget)
        settle(root)
        paused = len(view.folder_cards)
        assert paused < 120
        settle(root)
        assert len(view.folder_cards) == paused and view._load_id is None
        beat()
        host.pack(fill='both', expand=True)
        deadline = time.monotonic() + 12
        while len(view.folder_cards) < 120 and time.monotonic() < deadline:
            settle(root, .05)
        assert len(view.folder_cards) == 120
        assert len(beats) > 10
        ui_budget(max(b-a for a, b in zip(beats, beats[1:])), .4)
        settle(root)
        assert not view.loading_label.winfo_manager()
        view._parent_canvas.yview_moveto(1)
        settle(root)
        assert view._parent_canvas.yview()[0] > 0
        root.focus_force()
        view._parent_canvas.focus_set()
        settle(root)
        assert root.focus_get() is view._parent_canvas
        for key in ('Home', 'Next', 'Prior', 'Down', 'Up', 'End'):
            previous = view._parent_canvas.yview()[0]
            view._parent_canvas.event_generate(f'<KeyPress-{key}>')
            settle(root, .04)
            current = view._parent_canvas.yview()[0]
            if key == 'Home':
                assert current == 0
            elif key in ('Next', 'Down', 'End'):
                assert current > previous
            else:
                assert current < previous
        host.pack_forget()
        settle(root)
        host.pack(fill='both', expand=True)
        settle(root)
        assert view._parent_canvas.yview()[0] > 0
        # Exercise scale changes and teardown while callbacks are queued.
        ctk.set_widget_scaling(1.5)
        ctk.set_window_scaling(1.5)
        settle(root, .4)
        check_labels(root, view)
        view.destroy()
        settle(root)
        assert view._load_id is None
        assert view.folder_grid._layout_id is None
        replacement = PlacemarksView(host, {'placemarks': points})
        replacement.pack(fill='both', expand=True)
        settle(root, .05)
        replacement.destroy()
        settle(root)
        assert replacement._load_id is None
        assert not errors, errors
    finally:
        root.destroy()
        ctk.set_widget_scaling(1)
        ctk.set_window_scaling(1)
