"""Native test deadlines are independent of per-operation responsiveness."""
from types import SimpleNamespace
import subprocess
import sys

import pytest
import conftest


def item(**options):
    return SimpleNamespace(nodeid='tests/example.py::test_native',
                           get_closest_marker=lambda name: SimpleNamespace(kwargs=options))


@pytest.mark.parametrize('options, deadline, traceback', [
    ({}, 45, 20), ({'timeout': 120, 'traceback_timeout': 110}, 120, 110),
])
@pytest.mark.parametrize('platform', ['win32', 'darwin', 'linux'])
def test_native_deadlines_keep_fatal_capture(monkeypatch, options, deadline, traceback, platform):
    monkeypatch.delenv('PIPELINE_GUI_TEST_CHILD', raising=False)
    monkeypatch.setattr(conftest, 'sys', SimpleNamespace(platform=platform, executable=sys.executable,
                                                       stderr=sys.stderr))
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(conftest, 'run_gui', run)
    assert conftest.pytest_pyfunc_call(item(**options)) is True
    command, options = calls[0]
    assert command[1:3] == ['-X', 'faulthandler']
    expected_traceback = 0 if platform == 'win32' else traceback
    assert f'faulthandler_timeout={expected_traceback}' in command
    assert options['timeout'] == deadline
    assert options['env']['PIPELINE_GUI_TEST_CHILD'] == '1'


@pytest.mark.parametrize('text', [
    'invalid command name "old_callback"', 'Exception in Tkinter callback', 'TclError: stale widget',
])
def test_native_zero_exit_does_not_hide_tcl_errors(monkeypatch, text):
    monkeypatch.delenv('PIPELINE_GUI_TEST_CHILD', raising=False)
    monkeypatch.setattr(conftest, 'run_gui',
                        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, 'progress', text))
    with pytest.raises(pytest.fail.Exception, match='Tk callback error'):
        conftest.pytest_pyfunc_call(item())


def test_native_deadline_failure_keeps_progress_and_traceback(monkeypatch):
    monkeypatch.delenv('PIPELINE_GUI_TEST_CHILD', raising=False)
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'],
                                        output='completed_cycles: 40', stderr='native traceback')
    monkeypatch.setattr(conftest, 'run_gui', run)
    with pytest.raises(pytest.fail.Exception) as failure:
        conftest.pytest_pyfunc_call(item(timeout=120, traceback_timeout=110))
    assert '120 seconds' in str(failure.value)
    assert 'completed_cycles: 40' in str(failure.value)
    assert 'native traceback' in str(failure.value)


def test_native_traceback_must_precede_hard_deadline(monkeypatch):
    monkeypatch.delenv('PIPELINE_GUI_TEST_CHILD', raising=False)
    with pytest.raises(pytest.UsageError, match='traceback_timeout'):
        conftest.pytest_pyfunc_call(item(timeout=20, traceback_timeout=20))
