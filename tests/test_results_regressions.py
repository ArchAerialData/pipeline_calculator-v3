"""Exercise live result creation and the real corridor row actions."""
import time
from pathlib import Path
import pytest
import customtkinter as ctk


@pytest.mark.native_gui
def test_summary_created_inside_mainloop_settles(monkeypatch):
    from pipeline_calculator.gui.main_window import PipelineCalculatorGUI, messagebox
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView
    from pipeline_calculator.gui.layout import ResultPages
    ctk.ScalingTracker.get_window_dpi_scaling = classmethod(lambda cls, window: 1.25)
    app = PipelineCalculatorGUI()
    errors, samples, heartbeats = [], [], []
    monkeypatch.setattr(messagebox, 'showerror', lambda *a, **k: errors.append(a) or app.root.quit())
    app.root.report_callback_exception = lambda *a: errors.append(a) or app.root.quit()
    def results():
        app.state.current_results = {'analysis_complete': True, 'total_miles': 2344.563,
            'pipelines': [{}]*111, 'overlap_analysis': {'effective_total_miles': 2245.209,
                'savings_miles': 99.354, 'savings_percentage': 4.2, 'bundled_sections': [{}]*95}}
        app.show_results()
    def sample():
        pages = next(w for w in app.root.winfo_children() if isinstance(w, ResultPages))
        def find(node):
            for child in node.winfo_children():
                if isinstance(child, SummaryView):
                    return child
                found = find(child)
                if found:
                    return found
        summary = find(pages)
        samples.append((summary.winfo_width(), summary.winfo_height(), summary.original.winfo_width()))
    def pulse():
        heartbeats.append(time.monotonic())
        app.root.after(100, pulse)
    try:
        app.root.after(250, results)
        app.root.after(500, pulse)
        app.root.after(2200, sample)
        app.root.after(3400, sample)
        app.root.after(3600, app.root.quit)
        app.run()
        assert not errors, errors
        assert len(samples) == 2 and samples[0] == samples[1], samples
        assert len(heartbeats) >= 20, 'Layout must not starve the main event loop'
    finally:
        app.close()


@pytest.mark.native_gui
def test_corridor_buttons_scroll_page_and_launch_exact_kml(monkeypatch):
    from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable
    from pipeline_calculator.gui.dialogs.corridor_dialog import CorridorDialog
    from pipeline_calculator.gui.actions import open_kml_action as action
    ctk.ScalingTracker.get_window_dpi_scaling = classmethod(lambda cls, window: 1)
    root = ctk.CTk()
    root.geometry('1100x600')
    launched, dialogs, errors = [], [], []
    root.report_callback_exception = lambda *a: errors.append(a)
    monkeypatch.setattr(action, 'open_path', lambda path: launched.append((path, Path(path).read_text(encoding='utf-8'))))
    sections = [{'pipeline_1': f'Pipeline {i}', 'pipeline_2': 'Partner', 'bundled_length_miles': i+1,
                 'average_separation': 3, 'corridor_polygon': [(-100, 30), (-99.99, 30), (-99.99, 30.01), (-100, 30.01), (-100, 30)]}
                for i in range(45)]
    def open_corridor(section, index):
        dialogs.append(CorridorDialog(root, section, index))
    table = CorridorTable(root, sections, open_corridor)
    table.pack(fill='both', expand=True)
    def settle():
        until = time.monotonic()+.25
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)
        assert not errors, errors
    def activate(item, expected_index):
        table.tree.see(item)
        table.tree.xview_moveto(1)
        settle()
        button = table.row_buttons[item]
        assert button.winfo_viewable()
        # Deliver an actual native mouse click, including ttk press/release handling.
        button.event_generate('<ButtonPress-1>', x=12, y=12)
        button.event_generate('<ButtonRelease-1>', x=12, y=12)
        settle()
        assert dialogs[-1].done.wait(3)
        dialogs[-1]._poll()
        assert dialogs[-1].index == expected_index
        assert dialogs[-1].section is sections[expected_index-1]
        assert f'Pipeline {expected_index-1} + Partner' in launched[-1][1]
        assert launched[-1][0].endswith(f'_corridor_{expected_index:03d}.kml')
        assert dialogs[-1].outcome.status == 'requested'
        dialogs[-1].close()
    try:
        settle()
        assert all(table.tree.heading(c, 'anchor') == table.tree.column(c, 'anchor') for c in table.tree['columns'])
        activate(table.tree.get_children()[0], 1)
        activate(table.tree.get_children()[-1], 20)
        table.next_button.invoke()
        settle()
        activate(table.tree.get_children()[3], 24)
        root.geometry('500x400')
        settle()
        table.tree.xview_moveto(0)
        settle()
        assert not any(b.winfo_ismapped() for b in table.row_buttons.values())
        table.next_button.invoke()
        settle()
        activate(table.tree.get_children()[-1], 45)
        for item, button in table.row_buttons.items():
            if button.winfo_ismapped():
                x, y, width, height = table.tree.bbox(item, 'Action')
                assert y <= button.winfo_y() and button.winfo_y()+button.winfo_height() <= y+height
    finally:
        root.destroy()
        for path, _ in launched:
            Path(path).unlink(missing_ok=True)
