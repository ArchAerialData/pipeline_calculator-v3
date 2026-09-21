"""Native action widths follow their text without wasting a row of results."""
import math
import json
import sys

import customtkinter as ctk
import pytest

from pipeline_calculator.gui.layout import ActionBar, ResultPages
from pipeline_calculator.gui.tabs.summary_tab import SummaryView
from pipeline_calculator.gui.window import fitted_geometry
from test_results_header import assert_inside, render
from test_repair_ui import source, workflow
from test_state_breakdown_ui import descendants
from test_ui_lifecycle import settle


def viewport_receipt(root, summary, bar, header):
    widgets = [('root', root), ('actions', bar), ('context', header)]
    parent = summary._parent_canvas
    while parent is not root:
        widgets.append((type(parent).__name__, parent))
        parent = parent.master
    return {
        'tk': root.tk.call('info', 'patchlevel'),
        'screen': [root.winfo_screenwidth(), root.winfo_screenheight()],
        'widget_scale': ctk.ScalingTracker.get_widget_scaling(bar),
        'window_scale': ctk.ScalingTracker.get_window_scaling(root),
        'allocations': [dict(name=name, path=str(widget), visible=bool(widget.winfo_viewable()),
                             width=widget.winfo_width(), height=widget.winfo_height(),
                             requested_width=widget.winfo_reqwidth(), requested_height=widget.winfo_reqheight())
                        for name, widget in widgets],
    }


@pytest.mark.native_gui
# CTk's macOS backend always returns DPI=1: Aqua already uses scaled points.
# Injecting a Windows monitor factor there double-scales the native window and
# makes the window manager clamp the requested logical viewport. Exercise Aqua
# natively; simulated per-monitor DPI factors belong to the Windows backend.
@pytest.mark.parametrize('scale', [1, 2.5] if sys.platform == 'win32' else [None])
@pytest.mark.parametrize('action_font_size', [None, 16])
def test_actions_use_measured_width_and_leave_visible_scrollable_summary(monkeypatch, scale, action_font_size):
    if scale is not None:
        monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling', classmethod(lambda cls, window: scale))
    if action_font_size is not None:
        monkeypatch.setitem(ctk.ThemeManager.theme['CTkFont'], 'size', action_font_size)
    root = ctk.CTk()
    if scale is None:
        assert ctk.ScalingTracker.get_window_dpi_scaling(root) == 1
        scale = ctk.ScalingTracker.get_window_scaling(root)
    root.minsize(1, 1)
    root.maxsize(600, 300)
    root.geometry('600x300+16+16')
    flow, _ = workflow(root)
    flow.source = source(verified=True, can_save=True, report={'status': 'verified'})
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        header = render(root, flow)
        bar = next(w for w in root.winfo_children() if isinstance(w, ActionBar))
        for width in (600, 440, 600):
            root.geometry(f'{width}x300+16+16')
            settle(root, .25)
            factor = ctk.ScalingTracker.get_widget_scaling(bar)
            minimum = math.ceil(max(w._text_label.winfo_reqwidth() for w in bar.buttons) / factor + 24)
            columns = min(4, int(bar.winfo_width() / ((minimum + 10) * factor)))
            assert bar._columns == max(1, columns)
            for button in bar.buttons:
                assert_inside(button, bar)
                assert button.winfo_width() >= button._text_label.winfo_reqwidth() + 24 * factor - 1
            if columns == 4:
                assert len({button.winfo_y() for button in bar.buttons}) == 1
            assert_inside(header.repair_button if header._compact else header.repair_notice, header)
            short_summary = next(w for w in descendants(root) if isinstance(w, SummaryView))
            short_canvas = short_summary._parent_canvas
            receipt = viewport_receipt(root, short_summary, bar, header)
            print(json.dumps(receipt), flush=True)
            assert short_canvas.winfo_viewable() and short_canvas.winfo_height() > 1, receipt
            assert short_summary.inner.winfo_viewable()
            short_canvas.yview_moveto(1)
            settle(root)
            assert short_canvas.yview()[0] > 0
            short_canvas.yview_moveto(0)
            settle(root)
            assert short_summary.inner.winfo_viewable()

        # Platform fonts may legitimately need another button row. Give that
        # measured row its height before testing the same readable result area;
        # do not force four columns by squeezing text or reducing its padding.
        one_row = max(button.winfo_reqheight() for button in bar.buttons) / factor + 8
        height = math.ceil(300 + max(0, bar.winfo_reqheight() / factor - one_row))
        root.maxsize(600, height)
        root.geometry(f'600x{height}+16+16')
        settle(root, .3)
        summary = next(w for w in descendants(root) if isinstance(w, SummaryView))
        canvas = summary._parent_canvas
        title = next(w for w in descendants(summary) if isinstance(w, ctk.CTkLabel)
                     and w.cget('text') == 'Analysis Summary')
        assert canvas.winfo_viewable() and summary.inner.winfo_viewable()
        # When the actual window manager permits this viewport, the first
        # content line must be visible, not just an empty scroll-frame margin.
        if root.winfo_width() / factor >= 590 and root.winfo_height() / factor >= 300:
            assert_inside(title, canvas)
            assert canvas.winfo_height() >= summary.original.value.winfo_height()
            offset = summary.original.value.winfo_rooty() - summary.winfo_rooty()
            canvas.yview_moveto(offset / summary.winfo_reqheight())
            settle(root)
            assert_inside(summary.original.value, canvas)
        assert canvas.yview()[1] < 1
        canvas.yview_moveto(1)
        settle(root)
        assert canvas.yview()[0] > 0
        canvas.yview_moveto(0)
        settle(root)
        assert summary.inner.winfo_viewable()
        ctk.set_widget_scaling(1 / scale)
        ctk.set_window_scaling(1 / scale)
        root.maxsize(1000, 800)
        root.geometry('900x700+16+16')
        settle(root, .3)
        assert summary._outer_padding[0] == 16
        assert not errors, errors
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows per-monitor DPI uses physical pixels; Aqua uses native points')
def test_results_fit_actual_small_monitor_at_250_percent(monkeypatch):
    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling', classmethod(lambda cls, window: 2.5))
    root = ctk.CTk()
    width, height, _, _ = fitted_geometry((0, 0, 1024, 728), 2.5)
    assert (width, height) == (380, 230)
    root.minsize(1, 1)
    root.maxsize(width, height)
    root.geometry(f'{width}x{height}+16+16')
    flow, _ = workflow(root)
    flow.source = source(verified=True, can_save=True, report={'status': 'verified'})
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        header = render(root, flow)
        bar = next(w for w in root.winfo_children() if isinstance(w, ActionBar))
        summary = next(w for w in descendants(root) if isinstance(w, SummaryView))
        canvas = summary._parent_canvas
        print(json.dumps(viewport_receipt(root, summary, bar, header)), flush=True)
        assert bar._compact and bar._columns == 4
        assert [b.cget('text') for b in bar.buttons] == ['Export', 'Adjust', 'Import', 'Exit']
        for button in bar.buttons:
            assert_inside(button, root)
            assert button.winfo_width() >= button._text_label.winfo_reqwidth() + 24 * 2.5 - 1
        pages = next(w for w in root.winfo_children() if isinstance(w, ResultPages))
        assert_inside(pages.scope_selector, pages.navigation)
        assert_inside(header.repair_button, header)
        assert canvas.winfo_viewable() and canvas.winfo_height() >= summary.original.metric.winfo_height()
        offset = summary.original.metric.winfo_rooty() - summary.winfo_rooty()
        canvas.yview_moveto(offset / summary.winfo_reqheight())
        settle(root)
        assert_inside(summary.original.value, canvas)
        assert_inside(summary.original.unit, canvas)
        canvas.yview_moveto(1)
        settle(root)
        assert canvas.yview()[0] > 0
        # Restoring a normal viewport restores descriptive actions and spacing.
        ctk.set_widget_scaling(.4)
        ctk.set_window_scaling(.4)
        root.maxsize(1000, 800)
        root.geometry('900x700+16+16')
        settle(root, .4)
        assert not bar._compact and [b.cget('text') for b in bar.buttons] == list(bar.full_labels)
        assert summary._outer_padding[0] == 16
        assert summary._parent_frame.cget('corner_radius') == 12
        assert not errors, errors
    finally:
        flow.close()
        root.destroy()
