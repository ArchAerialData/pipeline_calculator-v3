"""Isolated native Windows layout probe; optionally capture this app's window only."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import customtkinter as ctk
from tkinter import ttk
from pipeline_calculator.gui.layout import ActionBar, ResultPages, WrappedLabel
from pipeline_calculator.gui.main_window import PipelineCalculatorGUI


def descendants(node):
    for child in node.winfo_children():
        yield child
        yield from descendants(child)


def run(scale, width, height, capture=None, legacy=False):
    # Exercise the actual CTk DPI path without changing the user's OS settings.
    ctk.ScalingTracker.get_window_dpi_scaling = classmethod(lambda cls, window: scale)
    if legacy:
        from pipeline_calculator_v3 import PipelineCalculatorGUI as GUI
    else:
        GUI = PipelineCalculatorGUI
    app = GUI()
    root = app.root
    errors = []
    root.report_callback_exception = lambda *args: errors.append(''.join(traceback.format_exception(*args)))
    root.geometry(f'{width}x{height}+{50 if capture else -8000}+100')
    # Keep validation windows offscreen instead of refitting them onto the desktop.
    root._display_changed = lambda event: None
    if root._fit_id:
        root.after_cancel(root._fit_id)
        root._fit_id = None
    user = ctypes.windll.user32
    user.GetParent.argtypes = [wintypes.HWND]
    user.GetParent.restype = wintypes.HWND
    hwnd = user.GetParent(root.winfo_id())

    def settle():
        until = time.monotonic() + .16
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)
        assert not errors, errors

    def check(page, target=None):
        target = target or root
        settle()
        if page in ('Pipelines', 'Overlap Analysis', 'Placemarks', 'Diagnostics'):
            selected = next(w for w in root.winfo_children() if isinstance(w, ResultPages)).pages[page]
            tree = next(w for w in descendants(selected) if isinstance(w, ttk.Treeview))
            assert tree.winfo_viewable(), (page, 'table hidden')
        for widget in descendants(target):
            if not widget.winfo_viewable():
                continue
            if isinstance(widget, ctk.CTkButton):
                x = widget.winfo_rootx() - target.winfo_rootx()
                y = widget.winfo_rooty() - target.winfo_rooty()
                assert x >= 0 and y >= 0, (page, widget.cget('text'), x, y)
                assert x + widget.winfo_width() <= target.winfo_width()+2, (page, widget.cget('text'), 'right')
                assert y + widget.winfo_height() <= target.winfo_height()+2, (page, widget.cget('text'), 'bottom')
                label = widget._text_label
                if label is not None:
                    assert label.winfo_reqwidth() <= widget.winfo_width()-8*scale, (page, widget.cget('text'), 'text clipped')
            if isinstance(widget, ttk.Treeview):
                assert widget.winfo_height() >= 42*scale, (page, 'table unusable', widget.winfo_height())
                assert widget.cget('xscrollcommand') and widget.cget('yscrollcommand')
                assert ttk.Style(widget).lookup(widget.cget('style'), 'background') == '#242424'
        if capture:
            from PIL import ImageGrab
            folder = Path(capture)
            folder.mkdir(parents=True, exist_ok=True)
            ImageGrab.grab(window=user.GetParent(target.winfo_id())).save(folder / f'{scale}-{width}x{height}-{page}.png')

    try:
        settle()
        assert root.cget('fg_color') == '#181818'
        dark = ctypes.c_int()
        dwm = ctypes.windll.dwmapi.DwmGetWindowAttribute
        dwm.argtypes = [wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD]
        assert dwm(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark)) == 0
        assert dark.value == 1, 'native titlebar is not dark'
        assert root.TkdndVersion
        check('import')
        data = {'total_miles': 153.716, 'total_meters': 247000,
                'pipelines': [{'OBJECTID': i, 'Name': 'Long pipeline name ' * 15,
                               'Shape_Length': 100, 'pipelinelength': 1} for i in range(50)],
                'placemarks': [{'Placemark_ID': '1', 'Name': 'Long name '*20, 'Count': 1}],
                'diagnostics': [{'level': 'warning', 'code': 'EXAMPLE', 'message': 'Long diagnostic '*50}],
                'overlap_analysis': {'effective_total_miles': 150, 'bundled_sections': [
                    {'pipeline_1': f'Pipeline {i}', 'pipeline_2': 'B', 'bundled_length_miles': 1,
                     'average_separation': 2} for i in range(45)]}}
        if legacy:
            app.current_results, app.current_file = data, 'Long filename ' * 30 + '.kmz'
        else:
            app.state.current_results, app.state.current_file = data, 'Long filename ' * 30 + '.kmz'
        opened = []
        if legacy:
            app.view_overlap_kml = lambda section, index: opened.append(index)
        else:
            app.view_overlap_corridor = lambda section, index: opened.append(index)
        app.show_results()
        pages = next(w for w in root.winfo_children() if isinstance(w, ResultPages))
        for name in pages.pages:
            pages.set(name)
            check(name)
            if name == 'Overlap Analysis':
                buttons = {w.cget('text'): w for w in descendants(pages.pages[name]) if isinstance(w, ctk.CTkButton)}
                buttons['Next'].invoke()
                buttons['View Corridor'].invoke()
                assert opened == [21]
                buttons['Next'].invoke()
                buttons['View Corridor'].invoke()
                assert opened == [21, 41]
        app.reanalyze()
        check('parameters')
        app._params_dialog.close()
        app.show_file_selection()
        check('return-import')
        from unittest.mock import patch
        from pipeline_calculator.gui.dialogs.corridor_dialog import CorridorDialog
        with patch.object(CorridorDialog, 'retry'):
            dialog = CorridorDialog(root, {}, 1)
        dialog.window.geometry(f'{width}x{height}+{60 if capture else -8000}+100')
        dialog.label.configure(text='Example viewer error and temporary file information. ' * 50)
        check('corridor-dialog', dialog.window)
        dialog.close()
        # Exercise the real caution layout with a controllable worker stub.
        from types import SimpleNamespace
        from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
        from pipeline_calculator.gui.state import AnalysisParameters
        import threading
        context = SimpleNamespace(workload_warning=lambda: 'Exceptionally dense geometry. ' * 80,
                                  snapshot=lambda: None)
        job = SimpleNamespace(job_id='layout', done=threading.Event(), state='waiting', context=context,
                              cancel=lambda: None)
        session = AnalysisSession(root, lambda job: None, SimpleNamespace(start=lambda *args: job))
        session.start('Extremely long input filename ' * 30 + '.kmz', AnalysisParameters())
        check('caution')
        assert session.continue_button.winfo_viewable()
        session.close()
        return {'scale': scale, 'logical_size': [width, height], 'legacy': legacy, 'status': 'passed'}
    finally:
        app.close()


if __name__ == '__main__':
    print(json.dumps(run(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
                         sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None,
                         len(sys.argv) > 5 and sys.argv[5] == 'legacy')))
