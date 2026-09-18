"""Explicit local packaging smoke mode; never used by ordinary application startup."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import traceback
import time


def _check_live_results(result, implementation='new'):
    """Exercise the packaged analysis-to-results transition under a real mainloop."""
    if implementation == 'legacy':
        from pipeline_calculator_v3 import PipelineCalculatorGUI
    else:
        from pipeline_calculator.gui.main_window import PipelineCalculatorGUI
    from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable
    from pipeline_calculator.gui.layout import ResultPages
    from pipeline_calculator.gui.tabs.summary_tab import SummaryView
    from tkinter import ttk
    app = PipelineCalculatorGUI()
    assert app.root.TkdndVersion
    errors, opened, rendered = [], [], []
    app.root.report_callback_exception = lambda *args: errors.append(''.join(traceback.format_exception(*args))) or app.root.quit()
    app.view_overlap_corridor = lambda section, index: opened.append((section, index))
    app.view_overlap_kml = app.view_overlap_corridor
    def show():
        if implementation == 'legacy':
            app.current_results = result
        else:
            app.state.current_results = result
        app.show_results()
        next(w for w in app.root.winfo_children() if isinstance(w, ResultPages)).set('Pipelines')
        rendered.append(True)
    def children(node):
        for child in node.winfo_children():
            yield child
            yield from children(child)
    def check():
        pages = next(w for w in app.root.winfo_children() if isinstance(w, ResultPages))
        tree = next(w for w in children(pages.pages['Pipelines']) if isinstance(w, ttk.Treeview))
        if tree.row_loader.rows is not None:
            app.root.after(50, check)
            return
        assert tree.heading('Placemark ID', 'text') == 'Placemark ID'
        assert [tree.set(item, 'Placemark ID') for item in tree.get_children()[:-1]] == [
            p['Placemark_ID'] for p in result['pipelines']]
        sections = (result.get('overlap_analysis') or {}).get('bundled_sections') or []
        if sections:
            table = next(w for w in children(app.root) if isinstance(w, CorridorTable))
            next(iter(table.row_buttons.values())).invoke()
            assert opened == [(sections[0], 1)]
        cycle()
    cycles = []
    def cycle():
        pages = next(w for w in app.root.winfo_children() if isinstance(w, ResultPages))
        pages.set('Pipelines')
        def return_to_summary():
            pages.set('Summary')
            deadline[0] = time.monotonic() + 1
            app.root.after(30, inspect_summary)
        app.root.after(30, return_to_summary)
    deadline = [None]
    def inspect_summary():
        summary = next(w for w in children(app.root) if isinstance(w, SummaryView))
        if not summary.original.value.winfo_viewable() and time.monotonic() < deadline[0]:
            app.root.after(10, inspect_summary)
            return
        assert summary.winfo_viewable() and summary.original.value.winfo_viewable()
        assert summary.original.value.cget('text') != ''
        summary.toggle.invoke()
        cycles.append(True)
        if len(cycles) < 20:
            app.root.after(30, cycle)
        else:
            app.root.quit()
    try:
        app.root.after(250, show)
        app.root.after(1800, check)
        app.root.after(15000, app.root.quit)
        app.root.mainloop()
        assert rendered and len(cycles) == 20 and not errors, errors
        return {'summary_returns': len(cycles), 'callback_errors': errors,
                'tk_patchlevel': app.root.tk.call('info', 'patchlevel')}
    finally:
        app.close()


def _check_geography_packaging(directory):
    """Exercise real bundled boundaries, native geometry and map roundtrips offline."""
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.core.geography import load_boundaries
    from pipeline_calculator.core.options import AnalysisOptions
    from pipeline_calculator.export.package import export_analysis_package

    boundaries = load_boundaries()
    assert len(boundaries.geometries) == 51
    for longitude, latitude, state in [(-157.8583, 21.3069, "HI"), (-149.9003, 61.2181, "AK"),
                                       (-97.7431, 30.2672, "TX"), (-77.0365, 38.8977, "DC")]:
        assert boundaries.states_at(longitude, latitude) == [state]
    fixture = Path(directory) / "state-crossing.kml"
    fixture.write_text(
        '<kml><Placemark><name>State crossing</name><LineString><coordinates>'
        '-101,36.49 -101,36.51</coordinates></LineString></Placemark></kml>', encoding="utf-8")
    analyzer = PipelineAnalyzer()
    result = analyzer.analyze_complete(fixture, options=AnalysisOptions(state_breakdown=True))
    geography = result["geography"]
    assert geography["status"] == "complete", geography.get("diagnostics")
    assert {state["state_code"] for state in geography["states"]} == {"TX", "OK"}
    assert geography["reconciliation"]["passed"]
    output = export_analysis_package(result, directory, fixture, include_json=True)
    combined = analyzer.analyze_complete(output / "Combined/analysis.kmz")
    assert abs(combined["total_meters"] - result["total_meters"]) <= .001
    for state in geography["states"]:
        mapped = analyzer.analyze_complete(output / f"States/{state['state_name']}/analysis.kmz")
        assert abs(mapped["total_meters"] - state["interior_meters"]) <= .001
    return {
        "boundary_jurisdictions": len(boundaries.geometries),
        "state_codes": sorted(state["state_code"] for state in geography["states"]),
        "boundary_vintage": boundaries.boundary_source.get("vintage"),
        "reconciliation_passed": True,
        "package_map_roundtrips": True,
    }


def _check_repair_packaging(directory):
    """Prove frozen repair imports, explicit approval and safe-source reimport."""
    from pipeline_calculator.gui.controllers.analysis_controller import analyze_file, RepairRequired
    from pipeline_calculator.gui.state import AnalysisParameters
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.export.xlsx import build_analysis_workbook

    original = Path(directory) / 'repair-input.kml'
    payload = (b'<kml xmlns="http://www.opengis.net/kml/2.2">'
               b'<Document xsi:schemaLocation="http://www.opengis.net/kml/2.2 schema.xsd">'
               b'<Placemark><LineString><coordinates>-100,40 -100,40.001</coordinates>'
               b'</LineString></Placemark></Document></kml>')
    original.write_bytes(payload)
    try:
        analyze_file(str(original), AnalysisParameters(), prepare_repair=True)
    except RepairRequired as decision:
        source = decision.source_session
    else:
        raise AssertionError('Invalid input did not await repair approval')
    assert not source.verified
    result = analyze_file(str(original), AnalysisParameters(), source_session=source, approve_repair=True)
    assert source.verified and result['input_repair']['status'] == 'verified'
    assert original.read_bytes() == payload
    saved = Path(directory) / 'repair-saved.kml'
    source.save(saved)
    reopened = PipelineAnalyzer().analyze_complete(saved)
    assert reopened['total_meters'] == result['total_meters']
    workbook = build_analysis_workbook(result)
    assert 'Analysis Details' in workbook.sheetnames
    workbook.save(Path(directory) / 'repair-report.xlsx')
    return {'approval_required': True, 'source_unchanged': True,
            'geometry_verified': True, 'saved_copy_roundtrip': True, 'provenance_exported': True}


def _check_corridor_packaging(directory):
    """Exercise actual buffered bends, a loop, separate pieces and clipped exports."""
    import math
    import xml.etree.ElementTree as ET
    from zipfile import ZipFile
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    from pipeline_calculator.core.geography import load_boundaries
    from pipeline_calculator.core.options import AnalysisOptions
    from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml
    from pipeline_calculator.export.package import export_analysis_package
    from pipeline_calculator.export.xlsx import build_analysis_workbook

    analyzer = PipelineAnalyzer()
    geod = analyzer.geod
    namespace = {'k': 'http://www.opengis.net/kml/2.2'}

    def geographic(x, y):
        return geod.fwd(-100, 40, math.degrees(math.atan2(x, y)), math.hypot(x, y))[:2]

    def write_source(name, paths):
        path = Path(directory) / name
        path.write_text('<kml><Document>' + ''.join(
            f'<Placemark><name>{i}</name><LineString><coordinates>' +
            ' '.join(f'{lon!r},{lat!r}' for lon, lat in points) +
            '</coordinates></LineString></Placemark>' for i, points in enumerate(paths)) +
            '</Document></kml>', encoding='utf-8')
        return path

    def read_polygons(root):
        def coordinates(node):
            return [tuple(map(float, point.split(',')[:2])) for point in node.text.split()]
        return [Polygon(coordinates(p.find('k:outerBoundaryIs/k:LinearRing/k:coordinates', namespace)),
                        [coordinates(h) for h in p.findall('k:innerBoundaryIs/k:LinearRing/k:coordinates', namespace)])
                for p in root.findall('.//k:Polygon', namespace)]

    metric_paths = ([[(0, 0), (300, 0), (300, 300), (0, 300), (0, 0)]] * 2 +
                    [[(1000, 0), (1000, 300), (1300, 300)]] * 2 +
                    [[(2000, 0), (2000, 500)], [(2012, 0), (2012, 500)]])
    path = write_source('corridor-shapes.kml', [[geographic(*p) for p in points] for points in metric_paths])
    result = analyzer.analyze_complete(path)
    assert result['analysis_complete'], result['diagnostics']
    # Frozen before activating the buffered builder. These remain sampled-overlap
    # expectations, independent of any polygon's area, radius or vertex count.
    assert abs(result['total_meters'] - 4599.9999803715655) <= .000001
    assert result['overlap_analysis']['savings_meters'] == 2285.0
    sections = result['overlap_analysis']['bundled_sections']
    assert [s['bundled_length_meters'] for s in sections] == [1195., 595., 495.]
    maps = []
    for index, section in enumerate(sections, 1):
        assert section['visualization_schema_version'] == 1 and section['visualization_status'] == 'ready'
        assert section['visualization_kind'] == 'qualified_path_buffer'
        assert section['visualization_metadata']['policy'] == 'qualified_path_buffer_v1'
        xml = ET.fromstring(build_overlap_corridor_kml(section, index))
        shapes = read_polygons(xml)
        assert shapes and all(shape.is_valid and not shape.is_empty for shape in shapes)
        assert not xml.findall('.//k:LineString', namespace) and not xml.findall('.//k:Point', namespace)
        maps.append(shapes)
    assert sum(len(shape.interiors) for shape in maps[0]) == 1
    assert not unary_union(maps[0]).covers(Point(geographic(150, 150)))
    assert not unary_union(maps[1]).covers(Point(geographic(1150, 150)))
    assert len(maps[2]) == 2
    workbook = build_analysis_workbook(result)
    assert workbook['Pipeline Overlap Analysis']['N1'].value == 'Corridor Map'
    assert 'Analysis Details' in workbook.sheetnames
    workbook.save(Path(directory) / 'corridor-report.xlsx')

    partner_lon = geod.fwd(-101, 36.5, 90, 12)[0]
    crossing = write_source('corridor-states.kml', [
        [(-101, 36.495), (-101, 36.505)], [(partner_lon, 36.495), (partner_lon, 36.505)]])
    state_result = analyzer.analyze_complete(crossing, options=AnalysisOptions(state_breakdown=True))
    assert state_result['analysis_complete'] and state_result['geography']['analysis_complete']
    assert abs(state_result['total_meters'] - 2219.3659186411564) <= .000001
    assert state_result['overlap_analysis']['savings_meters'] == 1105.0
    states = state_result['geography']['states']
    assert {state['state_code']: state['interior_savings_meters'] for state in states} == {'OK': 605., 'TX': 500.}
    package = export_analysis_package(state_result, directory, crossing, include_json=True)
    boundaries = load_boundaries()
    checked_parts = 0
    for state in states:
        state_map = package / f"States/{state['state_name']}/analysis.kmz"
        with ZipFile(state_map) as archive:
            shapes = read_polygons(ET.fromstring(archive.read('doc.kml')))
        assert shapes and all(shape.is_valid and shape.difference(boundaries.geometries[state['state_code']]).is_empty
                              for shape in shapes)
        checked_parts += len(shapes)
        reimported = analyzer.analyze_complete(state_map)
        assert abs(reimported['total_meters'] - state['interior_meters']) <= .001
    combined = analyzer.analyze_complete(package / 'Combined/analysis.kmz')
    assert abs(combined['total_meters'] - state_result['total_meters']) <= .001
    return ({'policy': 'qualified_path_buffer_v1', 'curved_geometry': True,
             'holes_preserved': True, 'multipart_preserved': True, 'polygon_only_preview': True,
             'state_containment': True, 'numeric_parity': True, 'map_roundtrips': True,
             'state_polygon_count': checked_parts}, result)


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
        # Both modes now exercise their real AppWindow below. A preliminary
        # throwaway Tk root leaves CTk's interpreter timers behind on teardown.
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
            geography_report = _check_geography_packaging(directory)
            repair_report = _check_repair_packaging(directory)
            corridor_report, corridor_result = _check_corridor_packaging(directory)
        ui_report = _check_live_results(result if real_input else corridor_result, implementation)
        icon=icon_path()
        assert icon is not None and icon.exists()
        assert (resource_root()/'README.md').is_file()
        report={'status':'passed','implementation':implementation,'frozen':bool(getattr(sys,'frozen',False)),
                'version':get_version(),'total_meters':result['total_meters'],'icon':icon.name,
                'tk_widgets':True,'dnd_loaded':True,
                'live_results': True, 'ui_reliability': ui_report, 'pipeline_count': len(result['pipelines']),
                'placemark_ids': [p['Placemark_ID'] for p in result['pipelines']],
                'original_miles': result['total_miles'],
                'geography': geography_report,
                'repair': repair_report,
                'corridors': corridor_report,
                'adjusted_miles': (result.get('overlap_analysis') or {}).get('effective_total_miles')}
    except Exception as exc:
        report={'status':'failed','implementation':implementation,'error':str(exc)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return int(report['status']!='passed')
