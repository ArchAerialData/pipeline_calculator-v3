from __future__ import annotations

import sys
import os
import subprocess
import math
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation.gui_process import run_gui
from scripts.validation.check_packaged_smoke import validate_tk_output


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Give each native Tk test its own interpreter and bounded lifetime.

    Destroyed Tk roots can retain cycles/callbacks. Sharing them with later
    worker-thread tests risks finalizing Tcl objects on the wrong thread.
    """
    marker = pyfuncitem.get_closest_marker('native_gui')
    if not marker or os.environ.get('PIPELINE_GUI_TEST_CHILD'):
        return None
    timeout = marker.kwargs.get('timeout', 45)
    traceback_timeout = marker.kwargs.get('traceback_timeout', 20)
    if not (isinstance(timeout, (int, float)) and math.isfinite(timeout)
            and isinstance(traceback_timeout, (int, float)) and math.isfinite(traceback_timeout)
            and 0 < traceback_timeout < timeout):
        raise pytest.UsageError('native_gui requires 0 < traceback_timeout < timeout')
    env = dict(os.environ, PIPELINE_GUI_TEST_CHILD='1')
    try:
        result = run_gui(
            [sys.executable, '-X', 'faulthandler', '-m', 'pytest', '-vv', '-s', '-o',
             f'faulthandler_timeout={traceback_timeout}', pyfuncitem.nodeid],
            cwd=REPO_ROOT, env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'Native GUI test exceeded {timeout} seconds:\n{exc.stdout}\n{exc.stderr}', pytrace=False)
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        validate_tk_output(result.stderr)
    except ValueError:
        pytest.fail(f'Native GUI test printed a Tk callback error:\n{result.stdout}\n{result.stderr}', pytrace=False)
    # Keep progress and structured measurements in pytest's captured output,
    # including on success; -s exposes them for retained acceptance logs.
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    return True

