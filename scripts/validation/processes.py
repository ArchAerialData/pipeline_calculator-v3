"""Bound local validation processes, including Windows venv launcher children."""
import os
import signal
import subprocess
import sys


def run_bounded(command, *, timeout, **kwargs):
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          **options, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                # The PID belongs to this invocation. Kill its tree before the
                # launcher exits, otherwise an underlying Python process can remain.
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=15)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=15)
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
