"""GIS import decisions preserve worker ownership and the shared desktop routes."""
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.importing.models import ImportDecisionRequired, ImportFailure


class CapturedSource:
    requires_repair = False
    original_path = 'captured.zip'
    display_name = 'captured.zip'

    def __init__(self, selected=(), *, calls=None, layers=None):
        self.selected_layers = list(selected)
        self.verified = bool(selected)
        self.calls = calls if calls is not None else []
        self.layers = layers or [
            {'id': 'pipes', 'dataset': 'survey/pipes.shp', 'name': 'Pipes',
             'geometry_type': 'LineString', 'feature_count': 2, 'crs': None, 'supported': True, 'reason': ''},
            {'id': 'sites', 'dataset': 'survey/sites.shp', 'name': 'Sites',
             'geometry_type': 'Point', 'feature_count': 3, 'crs': 'EPSG:4326', 'supported': True, 'reason': ''},
            {'id': 'area', 'dataset': 'survey/area.shp', 'name': 'Areas',
             'geometry_type': 'Polygon', 'feature_count': 1, 'crs': 'EPSG:4326', 'supported': False,
             'reason': 'Polygon boundaries are not pipeline centerlines.'},
        ]

    def fresh_parse(self, context=None):
        self.calls.append(('parse', threading.get_ident()))
        if not self.selected_layers:
            raise ImportDecisionRequired(self, 'Choose layers.', selection_required=True, missing_crs=['pipes'])
        return {'captured': self}

    def with_selection(self, ids, crs_overrides=None, context=None):
        self.calls.append(('select', threading.get_ident(), tuple(ids), dict(crs_overrides or {})))
        if crs_overrides and 'invalid' in crs_overrides.values():
            raise ImportFailure('The coordinate system is not recognized.', code='invalid_crs')
        return CapturedSource(ids, calls=self.calls, layers=self.layers)


def completed(job):
    assert job.done.wait(10), 'Import worker did not finish'
    job._thread.join(2)
    assert not job._thread.is_alive()
    return job


def test_import_decision_and_selection_use_captured_source_on_worker(monkeypatch):
    import pipeline_calculator.importing as importing
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    source = CapturedSource()
    acquired = []
    analyzed = []

    def prepare(path, context=None):
        acquired.append((path, threading.get_ident()))
        return source

    def analyze(self, parsed, **kwargs):
        analyzed.append(parsed['captured'])
        return {'analysis_complete': True}

    monkeypatch.setattr(importing, 'prepare_import', prepare, raising=False)
    monkeypatch.setattr(PipelineAnalyzer, 'analyze_parsed', analyze)
    controller = AnalysisController()
    first = completed(controller.start('captured.zip', AnalysisParameters()))
    assert first.state == 'import_required' and first.error is None and first.result is None
    assert first.source_session is source and first.import_decision.missing_crs == ['pipes']
    assert not analyzed
    draft = {'layer_ids': ['pipes'], 'crs_overrides': {'pipes': 'EPSG:4269'}}
    second = completed(controller.start('captured.zip', first.params, source_session=source, import_selection=draft))
    assert second.state == 'completed' and second.job_id != first.job_id
    assert len(acquired) == 1 and source.selected_layers == []
    assert analyzed == [second.source_session] and second.source_session.selected_layers == ['pipes']
    assert all(call[1] != threading.get_ident() for call in source.calls)
    assert acquired[0][1] != threading.get_ident()
    # A subsequent retry keeps the selected snapshot without reacquiring the path.
    third = completed(controller.start('captured.zip', first.params, source_session=second.source_session))
    assert third.state == 'completed' and len(acquired) == 1


def test_invalid_crs_preserves_discovery_for_recovery(monkeypatch):
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    monkeypatch.setattr(PipelineAnalyzer, 'analyze_parsed', lambda *args, **kwargs: pytest.fail('Invalid import reached analysis'))
    source = CapturedSource()
    job = completed(AnalysisController().start('captured.zip', AnalysisParameters(), source_session=source,
        import_selection={'layer_ids': ['pipes'], 'crs_overrides': {'pipes': 'invalid'}}))
    assert job.state == 'failed' and isinstance(job.error, ImportFailure)
    assert job.source_session is source and source.selected_layers == []


@pytest.mark.parametrize('suffix', ['.kml', '.zip'])
def test_ordinary_kml_and_kml_zip_retry_preserve_captured_source(tmp_path, suffix):
    import zipfile
    from pipeline_calculator.gui.repair_ui import RepairWorkflow
    from pipeline_calculator.importing import prepare_import
    kml = ('<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Pipe</name>'
           '<LineString><coordinates>-100,40 -99.999,40</coordinates></LineString>'
           '</Placemark></Document></kml>')
    path = tmp_path / ('ordinary' + suffix)
    if suffix == '.zip':
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('doc.kml', kml)
    else:
        path.write_text(kml, encoding='utf-8')
    source = prepare_import(path)
    assert source.approved and not source.requires_repair and not source.verified
    expected = source.fresh_parse().pipelines
    workflow = RepairWorkflow(None, set_busy=lambda busy: None, on_return=lambda: None,
                              on_replace=lambda path: None, on_resume=lambda *args, **kwargs: None)
    workflow.source = source
    path.write_bytes(b'The path changed after import.')
    retained = workflow.session_for(path)
    assert retained is source and retained.fresh_parse().pipelines == expected
    retried = completed(AnalysisController().start(str(path), AnalysisParameters(), source_session=retained))
    assert retried.state == 'completed' and len(retried.result['pipelines']) == 1
    assert not workflow.verified, 'Ordinary KML reuse must not show a repaired-file notice'
    workflow.retire_source()
    assert workflow.session_for(path) is None


@pytest.mark.native_gui
@pytest.mark.parametrize('declared_crs', [None, 'LOCAL_CS["Unknown",UNIT["metre",1]]'])
def test_import_panel_requires_selection_and_crs_then_resumes_same_snapshot(declared_crs):
    import customtkinter as ctk
    from pipeline_calculator.gui.repair_ui import RepairWorkflow
    from test_ui_lifecycle import settle
    root = ctk.CTk()
    root.geometry('640x480')
    calls, errors = [], []
    root.report_callback_exception = lambda *args: errors.append(args)
    flow = RepairWorkflow(root, set_busy=lambda busy: calls.append(('busy', busy)),
        on_return=lambda: calls.append(('return',)), on_replace=lambda path: calls.append(('replace', path)),
        on_resume=lambda *args, **kwargs: calls.append(('resume', args, kwargs)))
    source = CapturedSource()
    source.layers[0]['crs'] = declared_crs
    job = SimpleNamespace(source_session=source, state='import_required', error=None,
                          file_path='captured.zip', params=AnalysisParameters(), options=None,
                          import_decision=ImportDecisionRequired(source, 'Assign CRS.', missing_crs=['pipes']))
    try:
        assert flow.handle_done(job)
        settle(root, .3)
        panel = flow.panel
        assert len(panel.rows) == 3
        panel.select_all(True)
        assert [key for key, value in panel.selection_vars.items() if value.get()] == ['pipes', 'sites']
        panel.select_all(False)
        assert not any(value.get() for value in panel.selection_vars.values())
        panel.apply()
        assert 'at least one' in panel.error_label.cget('text')
        panel.selection_vars['pipes'].set(True)
        panel.apply()
        assert 'coordinate system' in panel.error_label.cget('text')
        panel.crs_vars['pipes'].set('EPSG:4269')
        panel.apply()
        panel.apply()  # A stale widget callback cannot submit a second job.
        resumed = [call for call in calls if call[0] == 'resume']
        assert len(resumed) == 1
        assert resumed[0][2]['source_session'] is source
        assert resumed[0][2]['import_selection'] == {'layer_ids': ['pipes'], 'crs_overrides': {'pipes': 'EPSG:4269'}}
        assert flow.session_for('captured.zip') is source
        assert not flow.verified, 'An ordinary GIS source must not acquire a repaired-file notice'
        # Worker-side CRS errors return to the same layer selection with the draft retained.
        failed = SimpleNamespace(**{**vars(job), 'state': 'failed', 'error': ImportFailure('Invalid CRS')})
        assert flow.handle_done(failed)
        assert flow.panel.title_label.cget('text') == 'Could not import this GIS file'
        flow.offer_import(failed)
        settle(root, .2)
        assert flow.panel.crs_vars['pipes'].get() == 'EPSG:4269'
        import tkinter as tk
        current_panel = flow.panel
        root.focus_force()
        tk.Misc.focus_set(current_panel.selection_widgets['sites'])
        current_panel.selection_widgets['sites'].event_generate('<space>')
        settle(root)
        assert current_panel.selection_vars['sites'].get()
        current_panel._tab()
        assert root.focus_get() is not None
        current_panel._escape()
        assert flow.panel is None and flow.session_for('captured.zip') is source
        assert not errors, errors
    finally:
        flow.close()
        root.destroy()


def test_crs_card_labels_are_bounded_without_reading_full_wkt():
    from pipeline_calculator.gui.import_ui import crs_label
    assert crs_label({'crs': 'PROJCRS["NAD83 / Texas South",' + 'x' * 10000}) == 'NAD83 / Texas South'
    assert crs_label({'crs': 'x' * 10000}) == 'x' * 177 + '…'
    assert crs_label({'crs': None}) == 'Missing'


@pytest.mark.native_gui
@pytest.mark.parametrize('implementation', ['modern', 'legacy'])
def test_gis_browse_drop_and_retry_share_import_route(tmp_path, monkeypatch, implementation):
    import customtkinter as ctk
    from pipeline_calculator.gui import main_window, preferences
    from pipeline_calculator.gui.window import AppWindow
    from pipeline_calculator.importing.formats import FILE_TYPES
    from test_state_breakdown_ui import descendants
    from test_ui_lifecycle import settle
    import pipeline_calculator_v3 as legacy
    monkeypatch.setattr(preferences, 'preferences_path', lambda: tmp_path / 'preferences.json')
    captures, drops, errors, pickers = [], {}, [], []
    path = str(tmp_path / 'Client café pipes.ZIP')
    source = CapturedSource(['pipes'])
    source.original_path = path

    class Session:
        def __init__(self, root, complete, *args):
            self.complete = complete
            self.job = SimpleNamespace(state='running', result=None, error=None, source_session=source, file_path=path)

        def start(self, selected, params, **kwargs):
            captures.append((selected, kwargs))

        def close(self):
            pass

        def finish(self):
            self.job.state = 'cancelled'
            self.complete(self.job)

    def picker(**kwargs):
        pickers.append(kwargs)
        return path

    monkeypatch.setattr(main_window, 'AnalysisSession', Session)
    monkeypatch.setattr(legacy, 'AnalysisSession', Session)
    monkeypatch.setattr(main_window.filedialog, 'askopenfilename', picker)
    monkeypatch.setattr(main_window.messagebox, 'showerror', lambda *args, **kwargs: errors.append(args))
    monkeypatch.setattr(AppWindow, 'dnd_bind', lambda root, event, callback, *args: drops.update({event: callback}))
    app = (main_window.PipelineCalculatorGUI if implementation == 'modern' else legacy.PipelineCalculatorGUI)()
    app.root.report_callback_exception = lambda *args: errors.append(args)
    try:
        app.browse_file()
        assert pickers[0]['filetypes'] == FILE_TYPES
        assert captures[0][0] == path
        app._analysis_session.finish()
        settle(app.root)
        retry = next(w for w in descendants(app.root) if isinstance(w, ctk.CTkButton)
                     and w.cget('text') == 'Retry selected file')
        retry.invoke()
        assert captures[-1][1]['source_session'] is source
        app._analysis_session.finish()
        for suffix in ('.ZIP', '.SHP', '.gpkg', '.geojson', '.json'):
            dropped = str(Path(path).with_suffix(suffix))
            drops['<<Drop>>'](SimpleNamespace(data='{' + dropped + '}'))
            assert captures[-1][0] == dropped
            app._analysis_session.finish()
        assert not errors, errors
    finally:
        app.close()


@pytest.mark.native_gui
def test_gis_point_cards_use_layers_without_inventing_folders():
    import customtkinter as ctk
    from pipeline_calculator.gui.tabs.placemarks_tab import PlacemarksView
    from test_ui_lifecycle import settle
    from test_state_breakdown_ui import descendants
    root = ctk.CTk()
    root.geometry('800x650')
    try:
        view = PlacemarksView(root, {'placemarks': [
            {'source_dataset': 'sites.gpkg', 'source_layer': 'Valves'},
            {'source_dataset': 'sites.gpkg', 'source_layer': 'Valves'},
        ]})
        view.pack(fill='both', expand=True)
        settle(root)
        texts = [widget.cget('text') for widget in descendants(view) if isinstance(widget, ctk.CTkLabel)]
        assert 'By layer' in texts
        assert 'GIS layer\nSource: sites.gpkg' in texts
        assert 'By subfolder' not in texts
        assert view.inventory['total'] == 2 and view.inventory['folder_count'] == 0
    finally:
        root.destroy()


@pytest.mark.native_gui
def test_gis_pipeline_table_shows_layer_scoped_ids_in_combined_and_state_views():
    import customtkinter as ctk
    from tkinter import ttk
    from pipeline_calculator.gui.tabs.pipelines_tab import create
    from test_ui_lifecycle import settle
    from test_state_breakdown_ui import descendants
    root = ctk.CTk()
    root.geometry('900x650')
    row = {'source_feature_id': 0, 'Name': 'Pipe', 'source_layer': 'Pipes',
           'source_dataset': 'survey.gpkg', 'Shape_Length': 1000., 'pipelinelength': .62,
           'interior_meters': 900., 'shared_allocation_meters': 100.}
    try:
        for state in (None, 'TX'):
            host = ctk.CTkFrame(root)
            host.pack(fill='both', expand=True)
            create(host, {'state_code': state, 'pipelines': [row], 'total_meters': 1000.,
                          'total_miles': .62, 'interior_meters': 900., 'shared_allocation_meters': 100.})
            settle(root)
            tree = next(widget for widget in descendants(host) if isinstance(widget, ttk.Treeview))
            assert tree['columns'][:4] == ('Feature ID', 'Name', 'Layer', 'Source dataset')
            values = tree.item(tree.get_children()[0], 'values')
            assert values[:4] == ('0', 'Pipe', 'Pipes', 'survey.gpkg')
            assert len(tree.get_children()) == 2
            host.destroy()
    finally:
        root.destroy()


@pytest.mark.native_gui(timeout=60, traceback_timeout=35)
@pytest.mark.parametrize(('filename', 'expected_lines'), [('RF_Lehman_Pipes.zip', 228), ('Verdun_Pipes.zip', 1530)])
def test_customer_zip_drop_uses_real_importer_and_retry_keeps_capture(tmp_path, monkeypatch, filename, expected_lines):
    import shutil
    import time
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.gui import main_window, preferences
    from pipeline_calculator.gui.window import AppWindow
    from test_ui_lifecycle import settle

    fixture = Path(__file__).parent / 'fixtures/gis/customer_zip' / filename
    selected = tmp_path / ('Client café ' + filename)
    shutil.copyfile(fixture, selected)
    monkeypatch.setattr(preferences, 'preferences_path', lambda: tmp_path / 'preferences.json')
    # Exercise real discovery, capture, GDAL reading, normalization and the Tk
    # worker boundary. Numerical overlap/state coverage belongs to backend tests.
    monkeypatch.setattr(PipelineAnalyzer, 'analyze_parsed', lambda self, parsed, **kwargs: {
        'analysis_complete': True, 'pipelines': parsed.pipelines, 'placemarks': parsed.placemarks})
    drops, errors, completed_views = {}, [], []
    monkeypatch.setattr(AppWindow, 'dnd_bind', lambda root, event, callback, *args: drops.update({event: callback}))
    monkeypatch.setattr(main_window.messagebox, 'showerror', lambda *args, **kwargs: errors.append(args))
    app = main_window.PipelineCalculatorGUI()
    app.root.report_callback_exception = lambda *args: errors.append(args)
    monkeypatch.setattr(app, 'show_results', lambda: completed_views.append(app.state.current_results))

    def wait_for_run():
        deadline = time.monotonic() + 25
        while app._processing and time.monotonic() < deadline:
            settle(app.root, .03)
        assert not app._processing, 'ZIP import did not finish or unexpectedly requested a decision'
        assert not errors, errors

    try:
        drops['<<Drop>>'](SimpleNamespace(data='{' + str(selected) + '}'))
        wait_for_run()
        assert len(completed_views) == 1 and len(completed_views[0]['pipelines']) == expected_lines
        assert app.state.current_file == str(selected)
        source = app.repair_workflow.source
        assert source is not None and not source.requires_repair
        selected.write_bytes(b'The original path changed after import.')
        app.process_file(str(selected), reuse_source=True)
        wait_for_run()
        assert len(completed_views) == 2 and len(completed_views[1]['pipelines']) == expected_lines
        assert app.repair_workflow.source is source
        assert not app.repair_workflow.verified
    finally:
        app.close()
