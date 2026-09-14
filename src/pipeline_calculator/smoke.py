"""Explicit local packaging smoke mode; never used by ordinary application startup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile


def _check_live_results(result):
    """Exercise the packaged analysis-to-results transition under a real mainloop."""
    from pipeline_calculator.gui.main_window import PipelineCalculatorGUI
    from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable
    from pipeline_calculator.gui.layout import ResultPages
    from tkinter import ttk
    app = PipelineCalculatorGUI()
    assert app.root.TkdndVersion
    errors, opened, rendered = [], [], []
    app.root.report_callback_exception = lambda *args: errors.append(str(args)) or app.root.quit()
    app.view_overlap_corridor = lambda section, index: opened.append((section, index))
    def show():
        app.state.current_results = result
        app.show_results()
        rendered.append(True)
    def children(node):
        for child in node.winfo_children():
            yield child
            yield from children(child)
    def check():
        pages = next(w for w in app.root.winfo_children() if isinstance(w, ResultPages))
        tree = next(w for w in children(pages.pages['Pipelines']) if isinstance(w, ttk.Treeview))
        assert tree.heading('Placemark ID', 'text') == 'Placemark ID'
        assert [tree.set(item, 'Placemark ID') for item in tree.get_children()[:-1]] == [
            p['Placemark_ID'] for p in result['pipelines']]
        sections = (result.get('overlap_analysis') or {}).get('bundled_sections') or []
        if sections:
            table = next(w for w in children(app.root) if isinstance(w, CorridorTable))
            next(iter(table.row_buttons.values())).invoke()
            assert opened == [(sections[0], 1)]
    try:
        app.root.after(250, show)
        app.root.after(1800, check)
        app.root.after(2200, app.root.quit)
        app.root.mainloop()
        assert rendered and not errors, errors
    finally:
        app.close()


def run(output_path, *, implementation='new'):
    from pipeline_calculator.versioning import get_version
    from pipeline_calculator.gui.resources import icon_path, resource_root
    from pipeline_calculator.export.xlsx import build_analysis_workbook
    from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml
    from pipeline_calculator.gui.controllers.analysis_session import AnalysisSession
    from pipeline_calculator.gui.dialogs.corridor_dialog import CorridorDialog
    from tkinterdnd2 import TkinterDnD
    import customtkinter as ctk
    if implementation == 'legacy':
        from pipeline_calculator_v3 import PipelineAnalyzer
    else:
        from pipeline_calculator.core.analyzer import PipelineAnalyzer
    output=Path(output_path)
    try:
        if implementation == 'legacy':
            root=TkinterDnD.Tk()
            root.withdraw()
            try:
                label=ctk.CTkLabel(root,text='Packaging check')
                label.pack()
                root.update_idletasks()
            finally:
                root.destroy()
        with tempfile.TemporaryDirectory(prefix='pipeline-smoke-') as directory:
            path=Path(directory)/'fixture.kml'
            path.write_text('<kml><Placemark><LineString><coordinates>-100,40 -100,40.001</coordinates></LineString></Placemark></kml>')
            # Optional real dataset is consumed only in explicit --smoke-test mode.
            real_input = os.environ.get('PIPELINE_SMOKE_INPUT')
            result=PipelineAnalyzer().analyze_complete(Path(real_input) if real_input else path)
            assert result['analysis_complete']
            if not real_input:
                assert len(result['pipelines']) == 1 and 110 < result['total_meters'] < 112
            workbook = build_analysis_workbook(result)
            sheet = workbook['Pipeline Length Analysis']
            assert sheet['A1'].value == 'Placemark ID'
            assert [sheet.cell(i+2, 1).value for i in range(len(result['pipelines']))] == [
                p['Placemark_ID'] for p in result['pipelines']]
            workbook.save(Path(directory)/'result.xlsx')
        if implementation == 'new':
            _check_live_results(result)
        icon=icon_path()
        assert icon is not None and icon.exists()
        assert (resource_root()/'README.md').is_file()
        report={'status':'passed','implementation':implementation,'frozen':bool(getattr(sys,'frozen',False)),
                'version':get_version(),'total_meters':result['total_meters'],'icon':icon.name,
                'tk_widgets':True,'dnd_loaded':True,
                'live_results': implementation == 'new', 'pipeline_count': len(result['pipelines']),
                'placemark_ids': [p['Placemark_ID'] for p in result['pipelines']],
                'original_miles': result['total_miles'],
                'adjusted_miles': (result.get('overlap_analysis') or {}).get('effective_total_miles')}
    except Exception as exc:
        report={'status':'failed','implementation':implementation,'error':str(exc)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')
