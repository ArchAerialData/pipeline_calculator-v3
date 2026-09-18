"""Execute the actual setup helper with an isolated native-command boundary."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows setup helper")


def _setup(tmp_path, *, fail_at=0, existing=True):
    script = tmp_path / "scripts/windows/setup_windows.ps1"
    script.parent.mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/pipeline_calculator_entry.py").touch()
    (tmp_path / "requirements.txt").touch()
    (tmp_path / "requirements-dev.txt").touch()
    sentinel = tmp_path / ".venv/preserved.txt"
    if existing:
        sentinel.parent.mkdir()
        sentinel.write_text("existing environment must not be removed")
    fake = tmp_path / "python.cmd"
    fake.write_text(
        '@echo off\nsetlocal\nset /a call_count=0\n'
        'if exist "%SETUP_TEST_COUNT%" set /p call_count=<"%SETUP_TEST_COUNT%"\n'
        'set /a call_count+=1\n'
        '>"%SETUP_TEST_COUNT%" echo %call_count%\n'
        'if "%call_count%"=="%SETUP_TEST_FAIL_AT%" exit /b 7\n'
        'if "%~1"=="-c" exit /b 0\n'
        'if "%~1"=="-m" if "%~2"=="venv" mkdir "%~3"\n'
        'exit /b 0\n'
    )
    # Redirect only the interpreter; leave the helper's checks/control flow intact.
    text = (ROOT / "scripts/windows/setup_windows.ps1").read_text().replace(
        '$Py = Join-Path $VenvDir "Scripts\\python.exe"', f"$Py = '{fake}'"
    )
    script.write_text(text)
    count = tmp_path / "calls.txt"
    env = {**os.environ, "SETUP_TEST_COUNT": str(count), "SETUP_TEST_FAIL_AT": str(fail_at)}
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-Python", str(fake)], capture_output=True, text=True, env=env, timeout=30,
    )
    return result, int(count.read_text()), sentinel


@pytest.mark.parametrize("fail_at,expected", [
    (1, "existing .venv must use Python 3.11"),
    (2, "Build tools installation failed"),
    (3, "Application dependencies installation failed"),
    (4, "Test dependencies installation failed"),
    (5, "Tkinter import failed"),
    (6, "CustomTkinter import failed"),
    (7, "TkinterDnD import failed"),
    (8, "Scientific stack import failed"),
    (9, "Installed dependencies are inconsistent"),
])
def test_existing_environment_failure_stops_without_false_success(tmp_path, fail_at, expected):
    result, calls, sentinel = _setup(tmp_path, fail_at=fail_at)
    assert result.returncode != 0
    assert expected in result.stderr
    assert "Setup complete." not in result.stdout
    assert calls == fail_at  # No install/import continues after the failed native process.
    assert sentinel.read_text() == "existing environment must not be removed"


@pytest.mark.parametrize("existing,expected_calls", [(True, 9), (False, 11)])
def test_success_requires_all_checks(tmp_path, existing, expected_calls):
    result, calls, _ = _setup(tmp_path, existing=existing)
    assert result.returncode == 0, result.stderr
    assert "Setup complete." in result.stdout
    assert calls == expected_calls
    assert (tmp_path / ".venv").exists()


@pytest.mark.parametrize("fail_at,expected", [
    (1, "Python 3.11 is required"),
    (2, "Virtual environment creation failed"),
    (3, "newly created .venv must use Python 3.11"),
])
def test_bootstrap_failure_stops_before_dependencies(tmp_path, fail_at, expected):
    result, calls, _ = _setup(tmp_path, existing=False, fail_at=fail_at)
    assert result.returncode != 0
    assert expected in result.stderr
    assert "Setup complete." not in result.stdout
    assert calls == fail_at
    if fail_at == 1:
        assert not (tmp_path / ".venv").exists()


def test_version_check_uses_real_python_version(tmp_path):
    """The native -c argument must retain its quotes under Windows PowerShell."""
    script = tmp_path / "scripts/windows/setup_windows.ps1"
    script.parent.mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/pipeline_calculator_entry.py").touch()
    (tmp_path / "requirements.txt").touch()
    text = (ROOT / "scripts/windows/setup_windows.ps1").read_text()
    # Exit immediately after the actual bootstrap check; never create/install a venv.
    text = text.replace('  & $BootstrapPython @PythonPrefix -m venv $VenvDir',
                        '  Write-Host "Version accepted"\n  exit 0\n  & $BootstrapPython @PythonPrefix -m venv $VenvDir')
    script.write_text(text)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
         "-Python", sys.executable], capture_output=True, text=True, timeout=30,
    )
    supported = sys.version_info[:2] == (3, 11)
    assert (result.returncode == 0) == supported, result.stderr
    assert ("Version accepted" in result.stdout) == supported
    assert not (tmp_path / ".venv").exists()


def test_broken_existing_environment_is_preserved(tmp_path):
    script = tmp_path / "scripts/windows/setup_windows.ps1"
    script.parent.mkdir(parents=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/pipeline_calculator_entry.py").touch()
    (tmp_path / "requirements.txt").touch()
    sentinel = tmp_path / ".venv/preserved.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("keep this environment")
    script.write_text((ROOT / "scripts/windows/setup_windows.ps1").read_text())
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "existing .venv has no Windows Python interpreter" in result.stderr
    assert "Setup complete." not in result.stdout
    assert sentinel.read_text() == "keep this environment"
