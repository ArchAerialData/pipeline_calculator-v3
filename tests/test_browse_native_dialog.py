"""Check the real Windows file picker without touching the user's desktop."""
import ctypes
from ctypes import wintypes as w
import os
import sys
import threading
import time

import pytest


pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='Windows native file picker')


def _window_api():
    user = ctypes.WinDLL('user32', use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    signatures = {
        'GetAncestor': ([w.HWND, w.UINT], w.HWND),
        'GetWindow': ([w.HWND, w.UINT], w.HWND),
        'GetWindowThreadProcessId': ([w.HWND, ctypes.POINTER(w.DWORD)], w.DWORD),
        'EnumThreadWindows': ([w.DWORD, callback_type, w.LPARAM], w.BOOL),
        'GetClassNameW': ([w.HWND, w.LPWSTR, ctypes.c_int], ctypes.c_int),
        'GetWindowTextW': ([w.HWND, w.LPWSTR, ctypes.c_int], ctypes.c_int),
        'IsWindowVisible': ([w.HWND], w.BOOL),
        'IsWindowEnabled': ([w.HWND], w.BOOL),
        'PostMessageW': ([w.HWND, w.UINT, w.WPARAM, w.LPARAM], w.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(user, name)
        function.argtypes, function.restype = arguments, result
    return user, callback_type


@pytest.mark.native_gui
@pytest.mark.parametrize('implementation', ['modern', 'legacy'])
def test_native_browse_keeps_app_visible_and_owns_modal_picker(tmp_path, monkeypatch, implementation):
    from pipeline_calculator.gui import main_window, preferences
    import pipeline_calculator_v3 as legacy

    monkeypatch.setattr(preferences, 'preferences_path', lambda: tmp_path / 'preferences.json')
    gui_class = main_window.PipelineCalculatorGUI if implementation == 'modern' else legacy.PipelineCalculatorGUI
    app = gui_class()
    processed, errors, observations = [], [], []
    app.root.report_callback_exception = lambda *args: errors.append(args)
    monkeypatch.setattr(app, 'process_file', processed.append)
    user, callback_type = _window_api()
    finished = threading.Event()
    observer = None
    try:
        app.root.update()
        # Tk's ID is the client window; the native dialog owns its outer window.
        root_hwnd = user.GetAncestor(app.root.winfo_id(), 2)  # GA_ROOT
        process_id = w.DWORD()
        gui_thread = user.GetWindowThreadProcessId(root_hwnd, ctypes.byref(process_id))
        assert process_id.value == os.getpid()
        assert user.IsWindowVisible(root_hwnd)

        def inspect_and_cancel():
            # No Tk calls from this thread. Enumeration is limited to this
            # test's GUI thread, with a second process check before any message.
            deadline = time.monotonic() + 10
            while not finished.is_set():
                dialogs = []

                @callback_type
                def visit(hwnd, _):
                    owner_pid = w.DWORD()
                    user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner_pid))
                    name = ctypes.create_unicode_buffer(256)
                    title = ctypes.create_unicode_buffer(256)
                    user.GetClassNameW(hwnd, name, len(name))
                    user.GetWindowTextW(hwnd, title, len(title))
                    if owner_pid.value == os.getpid() and name.value == '#32770':
                        dialogs.append((hwnd, title.value))
                    return True

                user.EnumThreadWindows(gui_thread, visit, 0)
                for hwnd, title in dialogs:
                    if title == 'Choose a KML or KMZ file' and user.IsWindowVisible(hwnd):
                        observations.append({
                            'root_visible': bool(user.IsWindowVisible(root_hwnd)),
                            'root_enabled': bool(user.IsWindowEnabled(root_hwnd)),
                            'owner': user.GetWindow(hwnd, 4),  # GW_OWNER
                        })
                        user.PostMessageW(hwnd, 0x0111, 2, 0)  # WM_COMMAND, IDCANCEL
                        return
                    if time.monotonic() >= deadline:
                        # Dismiss unexpected dialogs owned by this test only,
                        # so an assertion can explain a failed picker contract.
                        user.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
                if time.monotonic() >= deadline + 5:
                    return  # The native_gui subprocess also has a hard limit.
                finished.wait(.02)

        observer = threading.Thread(target=inspect_and_cancel, daemon=True)
        observer.start()
        app.browse_file()
        finished.set()
        observer.join(2)
        app.root.update()

        assert len(observations) == 1, 'The expected native file picker never became visible'
        assert observations[0] == {
            'root_visible': True,
            'root_enabled': False,
            'owner': root_hwnd,
        }, observations
        assert not processed, 'Cancelling Browse must not start analysis'
        assert user.IsWindowVisible(root_hwnd)
        assert user.IsWindowEnabled(root_hwnd)
        assert app.root.state() == 'normal'
        assert not errors
    finally:
        finished.set()
        if observer is not None:
            observer.join(2)
        app.close()
