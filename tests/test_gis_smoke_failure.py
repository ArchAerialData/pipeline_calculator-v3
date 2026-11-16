"""A missing packaged dependency must report a failure, not wait behind a modal."""
import json
import sys

from pipeline_calculator.smoke import run


def test_smoke_reports_initialization_import_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pipeline_calculator_v3', None)
    output = tmp_path / 'smoke.json'
    assert run(output, implementation='legacy') == 1
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['status'] == 'failed'
    assert report['implementation'] == 'legacy'
    assert 'pipeline_calculator_v3' in report['error']
    assert 'ModuleNotFoundError' in report['traceback']
