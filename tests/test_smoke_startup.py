"""The live-results smoke must begin after native window startup completes."""
import sys
import time
import tkinter as tk

import customtkinter as ctk
import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.smoke import _check_live_results


@pytest.mark.native_gui
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows CTk titlebar startup drains events while withdrawn')
@pytest.mark.parametrize('implementation', ['new', 'legacy'])
def test_smoke_waits_for_mapped_root_during_busy_titlebar_startup(tmp_path, monkeypatch, implementation):
    if implementation == 'legacy':
        from pipeline_calculator_v3 import PipelineCalculatorGUI
    else:
        from pipeline_calculator.gui.main_window import PipelineCalculatorGUI
    source = tmp_path / 'one-line.kml'
    source.write_text('<kml><Placemark><name>Startup probe</name><LineString><coordinates>'
                      '-100,35 -100,35.004</coordinates></LineString></Placemark></kml>', encoding='utf-8')
    result = PipelineAnalyzer().analyze_complete(source)
    startup_states = []
    rendered_states = []
    native_loop_entered = []
    original_titlebar = ctk.CTk._windows_set_titlebar_color
    original_mainloop = tk.Tk.mainloop
    original_show = PipelineCalculatorGUI.show_results

    def busy_titlebar(root, mode):
        if hasattr(root, '_display') and not root._window_exists:
            # A bounded event backlog reproduces slow startup without changing
            # any visibility/state method or replacing the actual titlebar code.
            # Previously, smoke timers ran inside this nested update and failed
            # while the root was deliberately withdrawn, before mainloop began.
            deadline = time.monotonic() + 4
            def busy_event():
                startup_states.append((root.state(), root.winfo_viewable()))
                if time.monotonic() < deadline:
                    time.sleep(.005)
                    root.after(0, busy_event)
            root.after(0, busy_event)
        return original_titlebar(root, mode)

    def native_mainloop(root, *args, **kwargs):
        native_loop_entered.append(True)
        return original_mainloop(root, *args, **kwargs)

    def show_results(app):
        rendered_states.append((bool(native_loop_entered), app.root.winfo_viewable()))
        return original_show(app)

    monkeypatch.setattr(ctk.CTk, '_windows_set_titlebar_color', busy_titlebar)
    monkeypatch.setattr(tk.Tk, 'mainloop', native_mainloop)
    monkeypatch.setattr(PipelineCalculatorGUI, 'show_results', show_results)
    report = _check_live_results(result, implementation)
    assert ('withdrawn', 0) in startup_states, 'Probe must exercise the real hidden startup interval'
    assert rendered_states == [(True, 1)], 'Results should be rendered once, after native startup'
    assert report['summary_returns'] == 20
    assert report['callback_errors'] == []
