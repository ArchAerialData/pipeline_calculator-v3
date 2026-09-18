from __future__ import annotations

import json
from pathlib import Path
import plistlib
import subprocess

import pytest

from scripts.validation import check_packaged_smoke as smoke


def passed_report(implementation):
    return {
        "status": "passed", "implementation": implementation, "frozen": True, "version": "4.16-test",
        "ui_reliability": {"summary_returns": 20, "callback_errors": []},
        "geography": {"boundary_jurisdictions": 51, "state_codes": ["OK", "TX"],
                      "reconciliation_passed": True, "package_map_roundtrips": True},
        "repair": {key: True for key in ('approval_required', 'source_unchanged',
                   'geometry_verified', 'saved_copy_roundtrip', 'provenance_exported')},
        "corridors": {'policy': 'qualified_path_buffer_v1', **{key: True for key in (
            'curved_geometry', 'holes_preserved', 'multipart_preserved', 'polygon_only_preview',
            'state_containment', 'numeric_parity', 'map_roundtrips')}},
    }


@pytest.mark.parametrize('stderr', ['invalid command name "123after"',
                                    'Exception in Tkinter callback', '_tkinter.TclError: missing widget'])
def test_tcl_callback_errors_cannot_pass_packaging(stderr):
    with pytest.raises(ValueError, match='Tk callback error'):
        smoke.validate_tk_output(stderr)


def test_resolves_versioned_inner_macos_binary_from_plist(tmp_path):
    app = tmp_path / "Pipeline_Calculator.app"
    binary = app / "Contents/MacOS/Pipeline Calculator v4.16-test"
    binary.parent.mkdir(parents=True)
    binary.touch()
    with (app / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleExecutable": binary.name}, stream)
    assert smoke.resolve_executable(app) == binary
    with (app / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleExecutable": "../unexpected"}, stream)
    with pytest.raises(ValueError, match="CFBundleExecutable"):
        smoke.resolve_executable(app)


def test_offline_frozen_gate_runs_both_implementations_and_saves_reports(tmp_path, monkeypatch):
    artifact = tmp_path / "application.exe"
    artifact.touch()
    calls = []
    monkeypatch.setenv("PROJ_NETWORK", "ON")
    monkeypatch.setenv("PIPELINE_SMOKE_INPUT", "unexpected-large-file.kmz")
    monkeypatch.setenv("PIPELINE_GUI_TEST_MODE", "interactive")

    def launch(command, *, timeout, cwd, env):
        calls.append((command, timeout, cwd, env))
        Path(command[-1]).write_text(json.dumps(passed_report(env["PIPELINE_CALCULATOR_IMPL"])))
        return subprocess.CompletedProcess(command, 0, "output", "diagnostics")

    monkeypatch.setattr(smoke, "run_gui", launch)
    output = tmp_path / "reports"
    result = smoke.check_packaged_smoke(artifact, output, expected_version="4.16-test")
    assert result["status"] == "passed"
    assert [call[3]["PIPELINE_CALCULATOR_IMPL"] for call in calls] == ["new", "legacy"]
    for command, timeout, cwd, environment in calls:
        assert command[:2] == [str(artifact), "--smoke-test"]
        assert timeout == 90 and cwd == artifact.parent
        assert environment["PROJ_NETWORK"] == "OFF"
        assert environment["PIPELINE_GUI_TEST_MODE"] == "isolated"
        assert "PIPELINE_SMOKE_INPUT" not in environment
    assert json.loads((output / "summary.json").read_text())["status"] == "passed"
    assert (output / "legacy.stdout.txt").read_text() == "output"
    assert (output / "new.stderr.txt").read_text() == "diagnostics"
    assert not list(output.glob(".packaged-smoke-*"))


@pytest.mark.parametrize("defect", ["source_only", "wrong_implementation", "bad_reconciliation", "bad_roundtrip",
                                  "wrong_jurisdictions", "wrong_states", "wrong_version", "failed_status", "missing_repair",
                                  'missing_corridors', 'wrong_corridor_policy', 'malformed_corridors'])
def test_report_validation_rejects_incomplete_evidence(defect):
    report = passed_report("new")
    if defect == "source_only":
        report["frozen"] = False
    elif defect == "wrong_implementation":
        report["implementation"] = "legacy"
    elif defect == "bad_reconciliation":
        report["geography"]["reconciliation_passed"] = False
    elif defect == "bad_roundtrip":
        report["geography"]["package_map_roundtrips"] = False
    elif defect == "wrong_jurisdictions":
        report["geography"]["boundary_jurisdictions"] = 50
    elif defect == "wrong_states":
        report["geography"]["state_codes"] = ["TX"]
    elif defect == "wrong_version":
        report["version"] = "4.15"
    elif defect == 'missing_repair':
        del report['repair']
    elif defect == 'missing_corridors':
        del report['corridors']
    elif defect == 'malformed_corridors':
        report['corridors'] = True
    elif defect == 'wrong_corridor_policy':
        report['corridors']['policy'] = 'legacy_rectangle'
    else:
        report["status"] = "failed"
    with pytest.raises(ValueError, match="Packaged smoke failed"):
        smoke.validate_report(report, "new", "4.16-test")


@pytest.mark.parametrize('flag', ['curved_geometry', 'holes_preserved', 'multipart_preserved',
                                 'polygon_only_preview', 'state_containment', 'numeric_parity', 'map_roundtrips'])
def test_each_new_corridor_smoke_flag_is_required(flag):
    report = passed_report('new')
    del report['corridors'][flag]
    with pytest.raises(ValueError, match=flag):
        smoke.validate_report(report, 'new', '4.16-test')


@pytest.mark.parametrize("failure", ["missing_report", "malformed_report", "nonzero_exit", "timeout"])
def test_failure_cannot_reuse_stale_reports_and_still_checks_other_implementation(tmp_path, monkeypatch, failure):
    artifact = tmp_path / "application.exe"
    artifact.touch()
    output = tmp_path / "reports"
    output.mkdir()
    (output / "new.json").write_text(json.dumps(passed_report("new")))
    calls = []

    def launch(command, *, timeout, cwd, env):
        implementation = env["PIPELINE_CALCULATOR_IMPL"]
        calls.append(implementation)
        if implementation == "legacy":
            Path(command[-1]).write_text(json.dumps(passed_report(implementation)))
            return subprocess.CompletedProcess(command, 0, "", "")
        if failure == "malformed_report":
            Path(command[-1]).write_text("not JSON")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout, output=b"partial output", stderr=b"timed out")
        return subprocess.CompletedProcess(command, 7 if failure == "nonzero_exit" else 0, "", "")

    monkeypatch.setattr(smoke, "run_gui", launch)
    result = smoke.check_packaged_smoke(artifact, output)
    assert result["status"] == "failed"
    assert calls == ["new", "legacy"]
    assert result["implementations"][0]["status"] == "failed"
    assert result["implementations"][1]["status"] == "passed"
    if failure == "malformed_report":
        assert (output / "new.json").read_text() == "not JSON"
    else:
        assert json.loads((output / "new.json").read_text())["status"] == "failed"
    if failure == "timeout":
        assert (output / "new.stdout.txt").read_text() == "partial output"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_cannot_disable_execution_bound(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        smoke.check_packaged_smoke(tmp_path / "unused.exe", tmp_path, timeout=timeout)
