"""Browse keeps its real app window visible during selection and failures."""
import tkinter as tk

import customtkinter as ctk
import pytest

from pipeline_calculator.gui import main_window, preferences
from test_state_breakdown_ui import descendants
from test_ui_lifecycle import settle


@pytest.mark.native_gui
@pytest.mark.parametrize('implementation', ['modern', 'legacy'])
def test_browse_keeps_app_visible_for_selection_cancel_and_errors(tmp_path, monkeypatch, implementation):
    import pipeline_calculator_v3 as legacy

    monkeypatch.setattr(preferences, 'preferences_path', lambda: tmp_path / 'preferences.json')
    gui = main_window.PipelineCalculatorGUI if implementation == 'modern' else legacy.PipelineCalculatorGUI
    app = gui()
    root = app.root
    errors, processed, unmapped, picker_calls = [], [], [], []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        settle(root, .4)
        original_state, original_geometry = root.state(), root.geometry()
        assert root.winfo_viewable()
        tk.Misc.bind(root, '<Unmap>', lambda event: unmapped.append(event) if event.widget is root else None,
                     add='+')
        browse = next(w for w in descendants(root) if isinstance(w, ctk.CTkButton)
                      and w.cget('text') == 'Browse Files')

        def visible():
            assert root.winfo_viewable()
            assert (root.state(), root.geometry()) == (original_state, original_geometry)

        def process(path):
            visible()
            processed.append(path)

        monkeypatch.setattr(app, 'process_file', process)
        dialog_errors = []

        def show_error(title, message, **kwargs):
            visible()
            assert kwargs['parent'] is root
            dialog_errors.append((title, message))

        monkeypatch.setattr(main_window.messagebox, 'showerror', show_error)
        selected = str(tmp_path / 'Client pipelines.kmz')
        for outcome in ('cancel', 'select', 'picker_error', 'cancel'):
            def pick(**kwargs):
                visible()  # Check while the picker is open, not just after it returns.
                assert kwargs['parent'] is root
                assert ('KMZ files', '*.kmz') in kwargs['filetypes']
                assert ('KML files', '*.kml') in kwargs['filetypes']
                picker_calls.append(outcome)
                if outcome == 'picker_error':
                    raise OSError('Picker unavailable')
                return selected if outcome == 'select' else ''

            monkeypatch.setattr(main_window.filedialog, 'askopenfilename', pick)
            browse.invoke()
            settle(root)
            visible()
            assert not unmapped, 'Browse must never hide and restore the application'

        assert picker_calls == ['cancel', 'select', 'picker_error', 'cancel']
        assert processed == [selected]
        assert dialog_errors == [('Error', 'Failed to browse file: Picker unavailable')]
        # Opening a picker still cannot interrupt a running analysis.
        app._processing = True
        browse.invoke()
        assert len(picker_calls) == 4
        assert not errors, errors
    finally:
        app.close()
