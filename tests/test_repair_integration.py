"""Repair approval, retained-source analysis and export integration contracts."""
from __future__ import annotations

import json
from pathlib import Path
import threading

import pytest

from pipeline_calculator.core.options import AnalysisOptions
from pipeline_calculator.gui.controllers.analysis_controller import AnalysisController, analyze_file
from pipeline_calculator.gui.state import AnalysisParameters
from pipeline_calculator.export.xlsx import build_analysis_workbook
from pipeline_calculator.export.repair_provenance import validate_input_repair
from pipeline_calculator.gui.actions.export_actions import export_results_to_path


def source_pair(tmp_path):
    good = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            '<Placemark><name>A &amp; B</name><LineString><coordinates>'
            '-101,36.49,1 -101,36.51,2</coordinates></LineString></Placemark>'
            '<Placemark><name>Parallel</name><LineString><coordinates>'
            '-101.00005,36.49,3 -101.00005,36.51,4</coordinates></LineString></Placemark>'
            '</Document></kml>')
    original = tmp_path / 'original.kml'
    original.write_text(good.replace('A &amp; B', 'A & B'), encoding='utf-8')
    valid = tmp_path / 'valid.kml'
    valid.write_text(good, encoding='utf-8')
    return original, valid


PARAMS = AnalysisParameters(detection_range=15, segment_length=100,
                            min_parallel_length=200, angular_tolerance=15)


def finish(job):
    assert job.done.wait(40), 'Worker failed to finish'
    job._thread.join(5)
    assert not job._thread.is_alive()
    assert job.error is None, job.error
    return job


@pytest.mark.parametrize('state_enabled', [False, True])
def test_approval_is_a_distinct_job_and_analysis_matches_known_good(tmp_path, state_enabled):
    original, valid = source_pair(tmp_path)
    original_bytes = original.read_bytes()
    options = AnalysisOptions(state_enabled)
    controller = AnalysisController()
    decision = finish(controller.start(str(original), PARAMS, options=options))
    assert decision.state == 'repair_required' and decision.result is None
    assert decision.context.snapshot().fraction < 1
    source = decision.source_session
    assert source.requires_repair and not source.verified
    with pytest.raises(Exception):
        source.fresh_parse()
    approved = finish(controller.start(str(original), PARAMS, options=options,
                                        source_session=source, approve_repair=True))
    assert approved.state == 'completed' and approved.job_id != decision.job_id
    assert approved.context.snapshot().fraction == 1
    expected = analyze_file(str(valid), PARAMS, options=options)
    actual = approved.result
    assert actual['total_meters'] == expected['total_meters']
    assert actual['overlap_analysis'] == expected['overlap_analysis']
    assert actual['pipelines'] == expected['pipelines']
    if state_enabled:
        assert [s['state_code'] for s in actual['geography']['states']] == ['OK', 'TX']
        for a, b in zip(actual['geography']['states'], expected['geography']['states']):
            assert a['total_meters'] == b['total_meters']
            assert a['adjusted_total_meters'] == b['adjusted_total_meters']
    assert source.verified and original.read_bytes() == original_bytes
    assert validate_input_repair(actual)['schema_version'] == 1
    assert json.loads(json.dumps(actual['input_repair'])) == actual['input_repair']
    workbook = build_analysis_workbook(actual)
    assert workbook.sheetnames.count('Analysis Details') == 1
    path = tmp_path / 'result.json'
    export_results_to_path(actual, str(path))
    assert json.loads(path.read_text())['input_repair'] == actual['input_repair']


def test_reanalysis_uses_fresh_frozen_records_after_original_changes(tmp_path):
    original, _ = source_pair(tmp_path)
    controller = AnalysisController()
    source = finish(controller.start(str(original), PARAMS)).source_session
    first = finish(controller.start(str(original), PARAMS, source_session=source, approve_repair=True))
    original.write_text('The sender replaced this file during the session.', encoding='utf-8')
    second = finish(controller.start(str(original), PARAMS, source_session=source))
    assert second.result['pipelines'] == first.result['pipelines']
    assert second.result['overlap_analysis'] == first.result['overlap_analysis']
    baseline = source.fresh_parse()
    baseline.pipelines[0]['coordinate_paths'][0][0] = (0, 0)
    assert source.fresh_parse().pipelines[0]['coordinate_paths'][0][0] != (0, 0)


def test_library_is_strict_by_default_and_explicit_approval_enables_repair(tmp_path):
    original, valid = source_pair(tmp_path)
    with pytest.raises(ValueError):
        analyze_file(str(original), PARAMS)
    result = analyze_file(str(original), PARAMS, approve_repair=True)
    assert result['input_repair']['status'] == 'verified'
    assert result['total_meters'] == analyze_file(str(valid), PARAMS)['total_meters']


def test_verified_source_survives_analysis_failure_and_cancellation(tmp_path, monkeypatch):
    from pipeline_calculator.core.analyzer import PipelineAnalyzer
    original, _ = source_pair(tmp_path)
    controller = AnalysisController()
    source = finish(controller.start(str(original), PARAMS)).source_session
    def fail(*args, **kwargs):
        raise RuntimeError('injected analyzer failure')
    monkeypatch.setattr(PipelineAnalyzer, 'analyze_parsed', fail)
    failed = controller.start(str(original), PARAMS, source_session=source, approve_repair=True)
    assert failed.done.wait(10)
    assert failed.state == 'failed' and source.verified
    assert failed.source_session is source and source.fresh_parse().pipelines
    entered, release = threading.Event(), threading.Event()
    def wait(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return {'analysis_complete': True}
    monkeypatch.setattr(PipelineAnalyzer, 'analyze_parsed', wait)
    cancelled = controller.start(str(original), PARAMS, source_session=source)
    assert entered.wait(10)
    cancelled.cancel()
    release.set()
    assert cancelled.done.wait(10)
    assert cancelled.state == 'cancelled' and cancelled.result is None
    assert cancelled.source_session is source and source.verified


@pytest.mark.parametrize('bad', [Path('private'), b'bytes', object(), float('nan'), (1, 2)])
def test_json_repair_provenance_cannot_be_hidden_by_default_str(tmp_path, bad):
    results = {'input_repair': {'schema_version': 1, 'evidence': bad}}
    output = tmp_path / 'report.json'
    with pytest.raises(ValueError):
        export_results_to_path(results, str(output))
    assert not output.exists()
    with pytest.raises(ValueError):
        build_analysis_workbook(results)


def test_repair_details_are_literal_and_ordinary_workbook_contract_unchanged():
    results = {'pipelines': [], 'overlap_analysis': None}
    assert build_analysis_workbook(results).sheetnames == ['Pipeline Length Analysis', 'Pipeline Overlap Analysis']
    results['input_repair'] = {'schema_version': 1, 'source_filename': '=2+2', 'status': 'verified'}
    sheet = build_analysis_workbook(results)['Analysis Details']
    matching = [c for row in sheet for c in row if c.value == '=2+2']
    assert len(matching) == 1 and matching[0].data_type == 's'
