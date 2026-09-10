"""Explicit local packaging smoke mode; never used by ordinary application startup."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


def run(output_path, *, implementation='new'):
    from pipeline_calculator.versioning import get_version
    from pipeline_calculator.gui.resources import icon_path, resource_root
    from pipeline_calculator.core.execution import ExecutionContext
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
            result=PipelineAnalyzer().analyze_complete(path,context=ExecutionContext())
            assert result['analysis_complete'] and len(result['pipelines'])==1
            assert 110 < result['total_meters'] < 112
            build_analysis_workbook(result).save(Path(directory)/'result.xlsx')
        icon=icon_path()
        assert icon is not None and icon.exists()
        assert (resource_root()/'README.md').is_file()
        report={'status':'passed','implementation':implementation,'frozen':bool(getattr(sys,'frozen',False)),
                'version':get_version(),'total_meters':result['total_meters'],'icon':icon.name,
                'tk_widgets':True,'dnd_loaded':True}
    except Exception as exc:
        report={'status':'failed','implementation':implementation,'error':str(exc)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')
