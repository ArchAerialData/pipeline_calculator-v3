"""Both GUI entrypoints preserve the preference across every input route."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.gui import preferences
from test_state_breakdown_ui import descendants, settle


@pytest.mark.native_gui
@pytest.mark.parametrize('implementation', ['modern', 'legacy'])
def test_persisted_options_browse_retry_drop_and_apply_cancel(tmp_path, monkeypatch, implementation):
    import customtkinter as ctk
    from pipeline_calculator.gui import main_window
    from pipeline_calculator.gui.window import AppWindow
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView
    import pipeline_calculator_v3 as legacy

    settings = tmp_path / 'preferences.json'
    preferences.save_state_breakdown(True, settings)
    monkeypatch.setattr(preferences, 'preferences_path', lambda: settings)
    captures, errors, drops = [], [], {}
    path = str(Path(__file__).parent / 'fixtures' / 'geography' / 'adamas_ng_pipeline_row.kmz')
    unsupported_result = PipelineAnalyzer().analyze_complete(path, options=AnalysisOptions(True))

    class Session:
        def __init__(self, root, complete, *args):
            self.complete = complete
            self.job = SimpleNamespace(state='running', result=None, error=None)

        def start(self, source, parameters, *, options=AnalysisOptions()):
            captures.append((source, options))

        def close(self):
            pass

        def finish(self, result=None):
            self.job.result = result
            self.job.state = 'completed' if result is not None else 'cancelled'
            self.complete(self.job)

    # Keep actual page wiring, preference binding and parameter dialogs. Only
    # replace the file picker, native drop transport and worker boundary.
    monkeypatch.setattr(main_window, 'AnalysisSession', Session)
    monkeypatch.setattr(legacy, 'AnalysisSession', Session)
    monkeypatch.setattr(main_window.filedialog, 'askopenfilename', lambda **kwargs: path)
    monkeypatch.setattr(main_window.messagebox, 'showerror', lambda *args, **kwargs: errors.append(args))
    monkeypatch.setattr(AppWindow, 'dnd_bind', lambda root, event, callback, *args: drops.update({event: callback}))
    gui_class = main_window.PipelineCalculatorGUI if implementation == 'modern' else legacy.PipelineCalculatorGUI
    app = gui_class()
    app.root.report_callback_exception = lambda *args: errors.append(args)
    try:
        def switch():
            return next(w for w in descendants(app.root) if isinstance(w, ctk.CTkSwitch))

        assert switch().get() == 1  # Loaded from a prior session's preferences.
        app.browse_file()
        assert captures == [(path, AnalysisOptions(True))]
        assert switch().cget('state') == 'disabled'
        app.process_file(path)
        assert len(captures) == 1  # Busy routes cannot start a second job.
        app._analysis_session.finish()
        assert switch().cget('state') == 'normal'
        retry = next(w for w in descendants(app.root) if isinstance(w, ctk.CTkButton)
                     and w.cget('text') == 'Retry selected file')
        retry.invoke()
        assert captures[-1] == (path, AnalysisOptions(True)) and len(captures) == 2
        app._analysis_session.finish()
        assert '<<Drop>>' in drops
        drops['<<Drop>>'](SimpleNamespace(data='{' + path + '}'))
        assert captures[-1] == (path, AnalysisOptions(True)) and len(captures) == 3
        app._analysis_session.finish(unsupported_result)
        settle(app.root)
        summary = next(w for w in descendants(app.root) if isinstance(w, SummaryView))
        texts = [w.cget('text') for w in descendants(summary) if isinstance(w, ctk.CTkLabel)]
        notice = next(text for text in texts if 'Analysis incomplete.' in text)
        assert 'No supported pipeline' in notice
        assert len(notice) < 500  # Per-polygon details stay out of the main notice.
        assert summary.adjusted.value.cget('text') == 'Not applicable'
        app.reanalyze()
        app._params_dialog._state_draft.set(False)
        app._params_dialog._cancel()
        assert preferences.load_state_breakdown(settings) is True
        assert app.state_preference.snapshot() == AnalysisOptions(True)
        assert len(captures) == 3
        app.reanalyze()
        app._params_dialog._state_draft.set(False)
        app._params_dialog._apply()
        assert preferences.load_state_breakdown(settings) is False
        assert captures[-1] == (path, AnalysisOptions(False)) and len(captures) == 4
        app._analysis_session.finish()
        assert not errors, errors
    finally:
        app.close()
    restarted = gui_class()
    try:
        assert restarted.state_preference.snapshot() == AnalysisOptions(False)
    finally:
        restarted.close()
