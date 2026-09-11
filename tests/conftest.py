from __future__ import annotations

import sys
import os
import subprocess
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Give each native Tk test its own interpreter and bounded lifetime.

    Destroyed Tk roots can retain cycles/callbacks. Sharing them with later
    worker-thread tests risks finalizing Tcl objects on the wrong thread.
    """
    if not pyfuncitem.get_closest_marker('native_gui') or os.environ.get('PIPELINE_GUI_TEST_CHILD'):
        return None
    env = dict(os.environ, PIPELINE_GUI_TEST_CHILD='1')
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-vv', '-s', '-o', 'faulthandler_timeout=20', pyfuncitem.nodeid],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=45,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f'Native GUI test exceeded 45 seconds:\n{exc.stdout}\n{exc.stderr}', pytrace=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return True

