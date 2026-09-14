"""Exercise desktop isolation, real mapped windows, and process-tree cleanup."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import wintypes as w
import json
import subprocess
import sys
import time

import pytest

from scripts.validation.gui_process import desktop_name, run_gui

pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows desktop isolation')


@pytest.fixture(autouse=True)
def isolated_mode(monkeypatch):
    monkeypatch.setenv('PIPELINE_GUI_TEST_MODE', 'isolated')


def test_real_window_is_mapped_without_reaching_input_desktop(tmp_path):
    marker, release = tmp_path / 'mapped.json', tmp_path / 'release'
    parent_desktop = desktop_name()
    code = '''
import ctypes,json,os,sys,tkinter as tk
from ctypes import wintypes as w
from pathlib import Path
from scripts.validation.gui_process import desktop_name
assert desktop_name() == os.environ['PIPELINE_TEST_DESKTOP']
root=tk.Tk()
root.title('Pipeline isolated focus regression')
root.geometry('640x480+20+20')
root.attributes('-topmost', True)
user=ctypes.windll.user32
user.GetParent.argtypes=[w.HWND]
user.GetParent.restype=w.HWND
def check():
    root.lift()
    root.focus_force()
    report={'desktop':desktop_name(),'expected':os.environ['PIPELINE_TEST_DESKTOP'],
            'state':root.state(),'viewable':bool(root.winfo_viewable()),
            'width':root.winfo_width(),'height':root.winfo_height(),
            'hwnd':user.GetParent(root.winfo_id())}
    Path(sys.argv[1]).write_text(json.dumps(report))
    poll()
def poll():
    if Path(sys.argv[2]).exists(): root.destroy()
    else: root.after(20,poll)
root.after(150,check)
root.mainloop()
'''
    user = ctypes.WinDLL('user32')
    user.GetForegroundWindow.restype = w.HWND
    callback_type = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    user.EnumWindows.argtypes = [callback_type, w.LPARAM]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_gui, [sys.executable, '-c', code, str(marker), str(release)], timeout=15)
        try:
            deadline = time.monotonic() + 10
            while not marker.exists():
                assert not future.done(), future.result()
                assert time.monotonic() < deadline, 'Test window never mapped'
                time.sleep(.02)
            report = json.loads(marker.read_text())
            assert report['desktop'] == report['expected'] != parent_desktop
            assert report['state'] == 'normal' and report['viewable']
            assert (report['width'], report['height']) == (640, 480)
            for _ in range(10):
                windows = []
                @callback_type
                def visit(hwnd, _):
                    windows.append(hwnd)
                    return True
                assert user.EnumWindows(visit, 0)
                assert report['hwnd'] not in windows, 'Test window leaked onto caller desktop'
                assert user.GetForegroundWindow() != report['hwnd'], 'Test window stole focus'
                time.sleep(.02)
        finally:
            release.touch()
        result = future.result()
    assert result.returncode == 0, result.stderr


def test_exit_code_unicode_arguments_and_working_directory(tmp_path):
    value = 'a path with spaces, "quotes", and café'
    code = "import os,sys; print(sys.argv[1]); print(os.getcwd()); print('failure detail',file=sys.stderr); sys.exit(7)"
    result = run_gui([sys.executable, '-c', code, value], cwd=tmp_path, timeout=10)
    assert result.returncode == 7
    assert result.stdout.splitlines() == [value, str(tmp_path)]
    assert 'failure detail' in result.stderr


def test_timeout_terminates_descendants(tmp_path):
    marker = tmp_path / 'child.pid'
    code = ("import subprocess,sys,time;from pathlib import Path;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            f"Path({str(marker)!r}).write_text(str(p.pid));print('child started',flush=True);time.sleep(60)")
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        run_gui([sys.executable, '-c', code], timeout=2)
    assert 'child started' in failure.value.output
    assert marker.exists()
    pid = int(marker.read_text())
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if handle:
        try:
            assert kernel.WaitForSingleObject(handle, 5000) == 0, 'Descendant survived timeout'
        finally:
            kernel.CloseHandle(handle)


def test_invalid_mode_does_not_fall_back_to_visible(monkeypatch):
    monkeypatch.setenv('PIPELINE_GUI_TEST_MODE', 'typo')
    with pytest.raises(ValueError, match='isolated or interactive'):
        run_gui([sys.executable, '-c', 'pass'], timeout=5)
