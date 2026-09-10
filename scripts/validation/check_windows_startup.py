"""Launch the real EXE normally and verify sustained native window visibility.

Unlike widget smoke tests, this does not call update(), ShowWindow(), or change
application state. It closes only the test window after observing it for 5 seconds.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import time


def check(exe):
    exe = Path(exe).resolve()
    user, kernel = ctypes.windll.user32, ctypes.windll.kernel32
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def windows():
        found = []

        @callback
        def visit(hwnd, _):
            pid = wintypes.DWORD()
            user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            handle = kernel.OpenProcess(0x1000, False, pid.value)
            if handle:
                try:
                    path, length = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
                    if kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(length)):
                        if Path(path.value) == exe:
                            title = ctypes.create_unicode_buffer(1024)
                            user.GetWindowTextW(hwnd, title, len(title))
                            if title.value.startswith('Pipeline Calculator v'):
                                found.append((hwnd, bool(user.IsWindowVisible(hwnd)), title.value))
                finally:
                    kernel.CloseHandle(handle)
            return True

        user.EnumWindows(visit, 0)
        return found

    assert not windows(), 'Close the existing instance of this test executable first'
    process = subprocess.Popen([str(exe)], cwd=exe.parent)
    first_visible = None
    observations = 0
    started = time.monotonic()
    try:
        while time.monotonic() - started < 25:
            assert process.poll() is None, f'EXE exited early: {process.returncode}'
            found = windows()
            visible = any(row[1] for row in found)
            if first_visible is not None:
                assert visible, 'Window disappeared after becoming visible'
                observations += 1
                if time.monotonic() - first_visible >= 5:
                    return {'status': 'passed', 'exe': str(exe), 'visible_seconds': 5,
                            'observations': observations, 'title': found[0][2]}
            elif visible:
                first_visible = time.monotonic()
            time.sleep(.1)
        raise AssertionError('Normal startup never produced a visible main window')
    finally:
        for hwnd, _, _ in windows():
            user.PostMessageW(hwnd, 0x0010, 0, 0)
        process.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('exe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.exe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
