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
    if scale == 2.5 and height == 288 and width in (408, 512):
        # Simulate 1020/1280x720 physical windows on larger developer monitors.
        root.minsize(1, 1)
        root.maxsize(width, height)
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
        if page in ('import', 'return-import'):
            from pipeline_calculator.gui.settings_panel import SettingsPanel
            panel = next(w for w in descendants(root) if isinstance(w, SettingsPanel))
            viewport = panel._parent_canvas
            overflow = panel.winfo_reqheight() > viewport.winfo_height() + 1
            assert bool(panel._scrollbar.winfo_manager()) == overflow, 'Scrollbar must match overflow'
            if not overflow:
                assert viewport.winfo_height() - panel.winfo_reqheight() <= 2*scale, 'Excess settings gap'
            assert panel._parent_frame.winfo_width() == panel._parent_frame.master.winfo_width(), [(str(w), w.winfo_width(), w.winfo_height()) for w in panel._parent_frame.master.winfo_children()]
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
                assert widget.winfo_height() >= 42*scale, (page, 'table unusable', widget.winfo_height(),
                    'root', root.winfo_width(), root.winfo_height(), 'scale', scale,
                    'ancestors', [(str(w), w.winfo_width(), w.winfo_height()) for w in
                                  (widget.master, widget.master.master, target)])
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
        data = {'total_miles': 153.716, 'total_meters': 247000, 'analysis_complete': True,
                'analysis_parameters': {'detection_range': 15.0, 'min_parallel_length': 200.0,
                                        'segment_length': 5.0, 'angular_tolerance': 15.0},
                'pipelines': [{'Placemark_ID': i, 'Name': 'Long pipeline name ' * 15,
                               'Shape_Length': 100, 'pipelinelength': 1} for i in range(50)],
                'placemarks': [{'Placemark_ID': '1', 'Name': 'Long name '*20, 'Count': 1}],
                'diagnostics': [{'level': 'warning', 'code': 'EXAMPLE', 'message': 'Long diagnostic '*50}],
                'overlap_analysis': {'effective_total_miles': 150, 'savings_miles': 3.716,
                                     'savings_percentage': 3.716/153.716*100, 'bundled_sections': [
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
        settle()
        # Navigation must adapt without changing the active page.
        if width >= 800:
            assert pages.tabs.winfo_viewable(), 'Tabs should fit at desktop widths'
        elif width <= 408:
            assert pages.selector.winfo_viewable(), 'Narrow windows need the menu'
        if scale == 1 and width == 1800:
            pages.set('Diagnostics')
            root.geometry('408x600')
            settle()
            assert pages.selector.winfo_viewable() and not pages.tabs.winfo_viewable()
            assert pages.pages['Diagnostics'].winfo_viewable()
            pages.selector.cget('command')('Summary')
            root.geometry(f'{width}x{height}')
            settle()
            assert pages.tabs.winfo_viewable() and not pages.selector.winfo_viewable()
            assert pages.pages['Summary'].winfo_viewable()
        for name in pages.pages:
            pages.set(name)
            assert pages.tabs.get() == pages.selector.get() == name
            check(name)
            if name == 'Summary':
                from pipeline_calculator.gui.tabs.summary_tab import SummaryView
                summary = next(w for w in descendants(pages.pages[name]) if isinstance(w, SummaryView))
                assert not summary.expanded and not summary.details.winfo_manager()
                if width >= 1280:
                    assert summary.original.grid_info()['row'] == summary.adjusted.grid_info()['row']
                elif width <= 640:
                    assert summary.original.grid_info()['row'] != summary.adjusted.grid_info()['row']
                    assert abs(summary.original.winfo_width() - summary.cards.winfo_width()) <= 2
                summary.toggle.invoke()
                settle()
                assert summary.details.winfo_manager() == 'pack'
                assert bool(summary._scrollbar.winfo_manager()) == (summary.winfo_reqheight() > summary._parent_canvas.winfo_height()+1)
                check('Summary-expanded')
                pages.set('Diagnostics')
                pages.set('Summary')
                assert summary.expanded, 'Tab switches must retain the disclosure state'
                summary.toggle.focus_force()
                settle()
                toggle_y = summary.toggle.winfo_rooty() - summary._parent_canvas.winfo_rooty()
                toggle_height, viewport_height = summary.toggle.winfo_height(), summary._parent_canvas.winfo_height()
                visible_height = min(toggle_y + toggle_height, viewport_height) - max(0, toggle_y)
                assert visible_height >= min(toggle_height, viewport_height)-2, (toggle_y, toggle_height, viewport_height)
                summary.toggle.event_generate('<Return>')
                settle()
                assert not summary.expanded, 'Return must toggle the disclosure'
                summary.toggle.event_generate('<KeyPress-space>')
                summary.toggle.event_generate('<KeyRelease-space>')
                settle()
                assert summary.expanded, 'Space must activate the disclosure'
                summary.toggle.invoke()
                summary._parent_canvas.yview_moveto(1)
                settle()
                # Check the real label boxes, including content below the fold.
                for label in descendants(summary):
                    if isinstance(label, ctk.CTkLabel) and label.winfo_viewable():
                        assert label._label.winfo_reqwidth() <= label.winfo_width()+2, ('Summary text clipped', label.cget('text'))
            if name == 'Overlap Analysis':
                from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable
                corridors = next(w for w in descendants(pages.pages[name]) if isinstance(w, CorridorTable))
                corridors.next_button.invoke()
                settle()
                next(iter(corridors.row_buttons.values())).invoke()
                assert opened == [21]
                corridors.next_button.invoke()
                settle()
                next(iter(corridors.row_buttons.values())).invoke()
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
        from pipeline_calculator.core.workload import warning_text
        context = SimpleNamespace(workload_warning=lambda: warning_text('About 754,586 analysis segments would be generated.'),
                                  snapshot=lambda: None)
        job = SimpleNamespace(job_id='layout', done=threading.Event(), state='waiting', context=context,
                              cancel=lambda: None)
        session = AnalysisSession(root, lambda job: None, SimpleNamespace(start=lambda *args: job))
        session.start('Q3 - WWM Pipelines.kmz', AnalysisParameters())
        check('caution')
        if session.guidance_columns == 1:
            for card in session.guidance_cards:
                assert abs(card.winfo_width() - session.guidance.winfo_width()) <= 2, \
                    'Single-column guidance must use the available width'
        assert session.continue_button.winfo_viewable()
        assert not session.bar.winfo_viewable(), 'No progress animation while waiting for a decision'
        if width >= 800 and height >= 600:
            assert session.frame.winfo_width() <= 842*scale
            # The structured notice includes separate guidance sections. It
            # should fit its content, while reserving the actions on short screens.
            assert session.frame.winfo_height() <= height*scale*.9+2
        if scale == 1 and width == 1800:
            from PIL import ImageGrab, ImageColor
            from pipeline_calculator.gui.modal import BACKDROP, SURFACE
            shot = ImageGrab.grab(window=user.GetParent(root.winfo_id())).convert('RGB')
            # Real pixels, not just configured colors: formerly these corners
            # showed the root's black background over a different sibling panel.
            for widget, background in [(session.frame, BACKDROP),
                                       (session.cancel_button, SURFACE),
                                       (session.continue_button, SURFACE)]:
                x = widget.winfo_rootx()-root.winfo_rootx()
                y = widget.winfo_rooty()-root.winfo_rooty()
                w, h = widget.winfo_width(), widget.winfo_height()
                for dx, dy in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
                    assert shot.getpixel((x+dx, y+dy)) == ImageColor.getrgb(background), (widget, dx, dy)
            assert not session.content._scrollbar.winfo_manager(), 'No scrollbar when the notice fits'
        # An unusually long warning and filename must still leave actions visible.
        context.workload_warning = lambda: 'Exceptionally dense geometry. ' * 80
        session.filename_label.configure(text='Extremely long input filename ' * 30 + '.kmz')
        settle()
        check('caution-long')
        if session.guidance_columns == 1:
            for card in session.guidance_cards:
                assert abs(card.winfo_width() - session.guidance.winfo_width()) <= 2, \
                    'Long guidance must not reserve an empty second column'
        session.close()
        return {'scale': scale, 'logical_size': [width, height], 'legacy': legacy, 'status': 'passed'}
    finally:
        app.close()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.validation.gui_process import isolate_probe
    isolate_probe()
    print(json.dumps(run(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
                         sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None,
                         len(sys.argv) > 5 and sys.argv[5] == 'legacy')))
