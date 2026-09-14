"""Verify both GUI implementations inside a built application, using offline data.

Windows launches use the existing isolated desktop runner. Other platforms use
its ordinary subprocess path. Fresh per-run report files prevent stale results
from making a crashed executable look successful.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.gui_process import run_gui


def resolve_executable(artifact):
    """Resolve a Windows/Linux executable or the actual binary named by a macOS app."""
    artifact = Path(artifact).resolve(strict=True)
    if artifact.is_dir():
        with (artifact / "Contents/Info.plist").open("rb") as stream:
            metadata = plistlib.load(stream)
        name = metadata.get("CFBundleExecutable")
        if not isinstance(name, str) or not name or name in (".", "..") or any(c in name for c in "/\\"):
            raise ValueError("The application bundle has an invalid CFBundleExecutable")
        artifact = artifact / "Contents/MacOS" / name
    if not artifact.is_file():
        raise FileNotFoundError(f"Packaged executable not found: {artifact}")
    return artifact


def validate_report(report, implementation, expected_version=None):
    if not isinstance(report, dict):
        raise ValueError("Smoke report must be a JSON object")
    checks = {
        "status": report.get("status") == "passed",
        "implementation": report.get("implementation") == implementation,
        "frozen executable": report.get("frozen") is True,
    }
    geography = report.get("geography")
    if not isinstance(geography, dict):
        geography = {}
    checks.update({
        "51 bundled jurisdictions": geography.get("boundary_jurisdictions") == 51,
        "Texas/Oklahoma crossing": geography.get("state_codes") == ["OK", "TX"],
        "geography reconciliation": geography.get("reconciliation_passed") is True,
        "package map roundtrips": geography.get("package_map_roundtrips") is True,
    })
    if expected_version is not None:
        checks["artifact version"] = report.get("version") == expected_version
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        detail = f"; application error: {report['error']}" if report.get("error") else ""
        raise ValueError("Packaged smoke failed: " + ", ".join(failed) + detail)


def _captured_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def check_packaged_smoke(artifact, output_directory, *, timeout=90, expected_version=None):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Smoke timeout must be finite and positive")
    executable = resolve_executable(artifact)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with tempfile.TemporaryDirectory(prefix=".packaged-smoke-", dir=output) as temporary:
        for implementation in ("new", "legacy"):
            report_path = Path(temporary) / f"{implementation}.json"
            environment = dict(os.environ, PROJ_NETWORK="OFF", PIPELINE_CALCULATOR_IMPL=implementation,
                               PIPELINE_GUI_TEST_MODE="isolated")
            environment.pop("PIPELINE_SMOKE_INPUT", None)
            outcome = {"implementation": implementation, "status": "failed"}
            stdout = stderr = ""
            report = None
            try:
                process = run_gui([str(executable), "--smoke-test", str(report_path)],
                                  timeout=timeout, cwd=executable.parent, env=environment)
                stdout, stderr = process.stdout, process.stderr
                outcome["exit_code"] = process.returncode
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                if process.returncode != 0:
                    raise ValueError(f"Packaged executable exited with code {process.returncode}")
                if report is None:
                    raise ValueError("Packaged executable did not create a fresh smoke report")
                validate_report(report, implementation, expected_version)
                outcome["status"] = "passed"
            except subprocess.TimeoutExpired as error:
                stdout, stderr = error.stdout, error.stderr
                outcome["error"] = f"Packaged executable exceeded {timeout:g} seconds"
            except (OSError, ValueError) as error:
                outcome["error"] = str(error)
            (output / f"{implementation}.stdout.txt").write_text(_captured_text(stdout), encoding="utf-8")
            (output / f"{implementation}.stderr.txt").write_text(_captured_text(stderr), encoding="utf-8")
            if report_path.is_file():
                # Preserve malformed application output too; summary.json records the failure.
                (output / f"{implementation}.json").write_bytes(report_path.read_bytes())
            else:
                (output / f"{implementation}.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
            outcomes.append(outcome)
    summary = {
        "status": "passed" if all(outcome["status"] == "passed" for outcome in outcomes) else "failed",
        "executable": str(executable), "proj_network": "OFF", "implementations": outcomes,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="Executable or macOS .app bundle")
    parser.add_argument("--output-directory", type=Path, default=REPO_ROOT / ".validation-output/packaged-smoke")
    parser.add_argument("--timeout", type=float, default=90, help="Maximum seconds per implementation")
    parser.add_argument("--expected-version", help="Require the packaged report to match build/version.json")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        result = check_packaged_smoke(args.artifact, args.output_directory,
                                     timeout=args.timeout, expected_version=args.expected_version)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Packaged smoke could not start: {error}\n")
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
