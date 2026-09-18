"""Repair UI regressions run on the repository's isolated native desktop."""
import threading
import time
from types import SimpleNamespace

import customtkinter as ctk
import pytest

from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.gui.repair_ui import RepairPanel, RepairWorkflow
from pipeline_calculator.gui.state import AnalysisParameters
from test_ui_lifecycle import settle


def source(**values):
    fields = dict(original_path='sample.kmz', display_name='sample.kmz', effective_format='kmz',
                  verified=False, requires_repair=True, can_save=False, save_unavailable_reason='',
                  report={'status': 'eligible', 'rules': ['missing_xsi_schema_namespace_v1']})
    fields.update(values)
    return SimpleNamespace(**fields)


def workflow(root):
    calls = []
    instance = RepairWorkflow(root, set_busy=lambda value: calls.append(('busy', value)),
        on_return=lambda: calls.append(('return',)), on_replace=lambda path: calls.append(('replace', path)),
        on_resume=lambda *args, **kwargs: calls.append(('resume', args, kwargs)))
    return instance, calls


@pytest.mark.native_gui
def test_repair_panel_actions_fit_focus_and_restore_at_high_dpi():
    ctk.ScalingTracker.get_window_dpi_scaling = classmethod(lambda cls, window: 1)
    root = ctk.CTk()
    root.geometry('640x480')
    original = ctk.CTkEntry(root)
    original.pack()
    errors = []
    actions = []
    root.report_callback_exception = lambda *args: errors.append(args)
    panel = None
    try:
        settle(root)
        original.focus_force()
        panel = RepairPanel(root, title='This file may be safely repairable',
            filename='nested/member/' * 30 + 'system.kml', message='Geometry will be verified before analysis.',
            details='Long details\n' * 100, actions=[('Repair & analyze', lambda: actions.append('repair')),
                ('Choose another file', lambda: None), ('Cancel', lambda: actions.append('cancel'))],
            on_cancel=lambda: actions.append('escape'))
        for scale, size in ((1, '640x480'), (1.25, '640x480'), (1.5, '640x480'), (2, '640x480'), (1, '390x844')):
            ctk.set_widget_scaling(scale)
            ctk.set_window_scaling(scale)
            width, height = (int(value) for value in size.split('x'))
            root.geometry(f'{round(width / scale)}x{round(height / scale)}')
            settle(root, .25)
            for button in panel.footer.buttons:
                assert button.winfo_viewable()
                assert button.winfo_rootx() >= root.winfo_rootx()
                assert button.winfo_rootx() + button.winfo_width() <= root.winfo_rootx() + root.winfo_width()
                assert button.winfo_rooty() + button.winfo_height() <= root.winfo_rooty() + root.winfo_height()
            panel._tab()
            assert root.focus_get() in panel.focus_controls
        import tkinter as tk
        tk.Misc.focus_set(panel.footer.buttons[0])
        panel.footer.buttons[0].event_generate('<Return>')
        tk.Misc.focus_set(panel.footer.buttons[2])
        panel.footer.buttons[2].event_generate('<space>')
        panel.footer.buttons[2].event_generate('<Escape>')
        assert actions == ['repair', 'cancel', 'escape']
        panel.toggle_details()
        settle(root)
        assert panel.detail_box.winfo_viewable()
        assert panel.footer.buttons[-1].winfo_viewable()
        panel.close()
        settle(root)
        assert not errors
        assert root.focus_get() == original._entry
        assert root.grab_current() is None
    finally:
        if panel is not None:
            panel.close()
        root.destroy()


@pytest.mark.native_gui
def test_approval_uses_captured_request_once_and_cancel_restores_input(monkeypatch):
    root = ctk.CTk()
    root.geometry('800x600')
    flow, calls = workflow(root)
    token = source()
    job = SimpleNamespace(source_session=token, state='repair_required', error=None,
        file_path='sample.kmz', params=AnalysisParameters(), options=AnalysisOptions(state_breakdown=False))
    try:
        assert flow.handle_done(job)
        settle(root)
        approve = flow.panel.footer.buttons[0]._command
        approve()
        approve()
        resumes = [call for call in calls if call[0] == 'resume']
        assert len(resumes) == 1
        assert resumes[0][1] == ('sample.kmz', job.params)
        assert resumes[0][2] == dict(options=job.options, source_session=token, approve_repair=True)
        flow.offer(job)
        settle(root)
        monkeypatch.setattr('pipeline_calculator.gui.repair_ui.filedialog.askopenfilename', lambda **kwargs: '')
        panel = flow.panel
        flow.choose_another()
        assert flow.panel is panel and flow.source is token
        assert root.grab_current() is panel.surface
        flow.cancel_decision()
        assert flow.source is None and flow.panel is None
        assert calls[-2:] == [('busy', False), ('return',)]
        assert root.grab_current() is None
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
def test_verified_notice_survives_scope_changes_and_failed_analysis():
    from pipeline_calculator.gui.pages.results_page import show
    from test_state_breakdown_ui import sample_results, descendants
    from tkinter import ttk
    root = ctk.CTk()
    root.geometry('1000x720')
    flow, calls = workflow(root)
    flow.source = source(verified=True, requires_repair=False, can_save=True,
                         report={'status': 'verified', 'rules': ['missing_xsi_schema_namespace_v1']})
    try:
        show(root, version='test', current_file='sample.kmz', current_results=sample_results(),
            on_export=lambda: None, on_reanalyze=lambda: None, on_new_file=lambda: None,
            on_exit=lambda: None, on_open_corridor=lambda *args: None, repair_workflow=flow)
        settle(root)
        notices = lambda: [w for w in descendants(root) if isinstance(w, ctk.CTkLabel)
                           and 'Geometry verified unchanged' in w.cget('text')]
        notice = notices()[0]
        selector = next(w for w in descendants(root) if isinstance(w, ttk.Combobox))
        for name in ('Texas', 'Combined', 'Texas'):
            selector.set(name)
            selector.event_generate('<<ComboboxSelected>>')
            settle(root)
            assert notices() == [notice] and notice.winfo_viewable()
        token = flow.source
        job = SimpleNamespace(source_session=token, state='cancelled', error=None)
        assert not flow.handle_done(job)
        assert flow.session_for('sample.kmz') is token
        assert flow.analysis_state == 'cancelled'
        from pipeline_calculator.gui.pages.file_select_page import show as show_input
        import tkinter as tk
        variables = [tk.StringVar(root, value=value) for value in ('15', '200', '5', '15')]
        show_input(root, title='Pipeline Calculator', detection_range_var=variables[0],
                   min_parallel_var=variables[1], segment_length_var=variables[2],
                   angular_tolerance_var=variables[3], on_browse=lambda: None,
                   on_file_selected=lambda path: None, retry_path='sample.kmz', repair_workflow=flow)
        root.geometry('640x480')
        settle(root, .3)
        assert len(notices()) == 1 and 'Analysis cancelled' in notices()[0].cget('text')
        for label in ('Browse Files', 'Retry selected file', 'Details', 'Save repaired copy…'):
            assert next(w for w in descendants(root) if isinstance(w, ctk.CTkButton)
                        and w.cget('text') == label).winfo_viewable()
        assert flow.session_for('other.kmz') is None
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
def test_client_request_copy_fallback_and_save_worker_cancellation(monkeypatch, tmp_path):
    root = ctk.CTk()
    root.geometry('640x480')
    flow, calls = workflow(root)
    try:
        request = 'Please ask the client to provide a new complete KMZ. Do not delete vertices.'
        class Failure(ValueError):
            findings = [{'code': 'invalid_xml'}]
            client_request = request
            category = 'source'
        flow.show_failure(Failure('XML is incomplete.'))
        settle(root)
        flow.panel.footer.buttons[0].invoke()
        assert root.clipboard_get() == request
        def fail_copy():
            import tkinter as tk
            raise tk.TclError('unavailable')
        monkeypatch.setattr(root, 'clipboard_clear', fail_copy)
        flow.panel.footer.buttons[0].invoke()
        assert flow.panel.detail_box is not None
        assert 'Could not copy' in flow.panel.message.cget('text')
        flow.cancel_decision()
        started = threading.Event()
        stopped = threading.Event()
        thread_ids = []
        def save(path, context=None):
            thread_ids.append(threading.get_ident())
            started.set()
            try:
                while not context.cancel_event.wait(.01):
                    context.check()
                context.check()
            finally:
                stopped.set()
        flow.source = source(verified=True, can_save=True, save=save, report={'status': 'verified'})
        monkeypatch.setattr('pipeline_calculator.gui.repair_ui.filedialog.asksaveasfilename',
                            lambda **kwargs: str(tmp_path / 'saved.kmz'))
        flow.save_copy()
        assert started.wait(1)
        flow.panel.footer.buttons[0].invoke()
        settle(root, .3)
        assert stopped.is_set()
        assert thread_ids != [threading.get_ident()]
        assert flow.panel is None and flow.source.verified
        assert calls[-1] == ('busy', False)
        started.clear()
        stopped.clear()
        flow.save_copy()
        assert started.wait(1)
        flow.close()
        assert stopped.wait(1)
        settle(root)
        assert flow.save_poll is None and flow.panel is None
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
def test_operational_failure_offers_support_report_not_client_reexport():
    from pipeline_calculator.parsers.repair import RepairFailure
    root = ctk.CTk()
    root.geometry('640x480')
    flow, calls = workflow(root)
    try:
        error = RepairFailure('The supported file size limit was reached.', category='limit')
        flow.show_failure(error)
        settle(root)
        labels = [button.cget('text') for button in flow.panel.footer.buttons]
        assert 'Copy diagnostic report' in labels
        assert 'Copy client request' not in labels
        assert 'ask the client' not in flow.panel.detail_text
        flow.panel.footer.buttons[0].invoke()
        assert 'limit' in root.clipboard_get()
        flow.cancel_decision()
        policy = RepairFailure('Encrypted archives are not supported.', category='policy', findings=[{
            'category': 'policy', 'code': 'encrypted', 'message': 'Encrypted archive.',
            'action': 'Provide a complete unencrypted KMZ export.'}])
        flow.show_failure(policy)
        assert flow.panel.footer.buttons[0].cget('text') == 'Copy client request'
        assert 'unencrypted' in flow.panel.detail_text
    finally:
        flow.close()
        root.destroy()


@pytest.mark.native_gui
@pytest.mark.parametrize('implementation', ['modern', 'legacy'])
def test_complete_repair_retry_and_explicit_reimport_in_both_entrypoints(tmp_path, monkeypatch, implementation):
    from pipeline_calculator.gui import main_window, preferences
    from pipeline_calculator.gui.window import AppWindow
    from test_state_breakdown_ui import descendants
    import pipeline_calculator_v3 as legacy
    selected = tmp_path / 'sample.kml'
    selected.write_text('<kml xmlns="http://www.opengis.net/kml/2.2"><Document xsi:schemaLocation="schema">'
                        '<Placemark><name>Source</name><LineString><coordinates>'
                        '-101,31 -101.0001,31</coordinates></LineString></Placemark></Document></kml>', encoding='utf-8')
    monkeypatch.setattr(preferences, 'preferences_path', lambda: tmp_path / 'prefs.json')
    errors, drops = [], {}
    monkeypatch.setattr(main_window.messagebox, 'showerror', lambda *args, **kwargs: errors.append(args))
    monkeypatch.setattr(AppWindow, 'dnd_bind', lambda root, event, callback, *args: drops.update({event: callback}))
    app = (main_window.PipelineCalculatorGUI if implementation == 'modern' else legacy.PipelineCalculatorGUI)()
    app.root.report_callback_exception = lambda *args: errors.append(args)
    def until(predicate):
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            settle(app.root, .02)
        assert predicate()
    try:
        app.process_file(str(selected))
        until(lambda: app.repair_workflow.panel is not None)
        first = app._analysis_session.job
        assert first.state == 'repair_required' and first.options.state_breakdown is False
        assert app._processing
        # Invalid drops must not open a second error panel over the decision.
        assert drops['<<Drop>>'](SimpleNamespace(data='one.kml two.kml')) == 'break'
        app.process_file('other.kml')
        assert app._analysis_session.job is first
        approve = app.repair_workflow.panel.footer.buttons[0]
        approve.invoke()
        until(lambda: not app._processing)
        result = app.state.current_results if implementation == 'modern' else app.current_results
        assert result['input_repair']['status'] == 'verified'
        assert len(result['pipelines']) == 1
        assert app._analysis_session.job is not first
        token = app.repair_workflow.source
        selected.write_text('<kml>new incomplete source', encoding='utf-8')
        # Retry analysis intentionally uses the retained snapshot, current options.
        app.state_preference.variable.set(False)
        app.process_file(str(selected), reuse_source=True)
        until(lambda: not app._processing)
        assert app.repair_workflow.source is token
        assert app._analysis_session.job.state == 'completed'
        # A new explicit import of the same pathname must inspect changed bytes.
        app.process_file(str(selected))
        until(lambda: app.repair_workflow.panel is not None)
        assert app._analysis_session.job.state == 'failed'
        assert app.repair_workflow.source is None
        assert any(button.cget('text') == 'Copy client request'
                   for button in app.repair_workflow.panel.footer.buttons)
        app.repair_workflow.cancel_decision()
        assert not app._processing
        assert not errors, errors
    finally:
        app.close()
