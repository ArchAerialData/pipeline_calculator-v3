"""Run real Windows GUI checks on an undisplayed desktop, without changing Tk.

No SwitchDesktop, ShowWindow, focus suppression, or visibility mocks. Native
windows remain mapped on their test desktop. Fail closed if isolation fails.
STARTUPINFO.lpDesktop is not exposed by Python's subprocess module, so keep
the small Win32 launcher here. Other platforms retain normal subprocess behavior.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid


class StartupInfo(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('lpReserved', w.LPWSTR), ('lpDesktop', w.LPWSTR),
                ('lpTitle', w.LPWSTR), ('dwX', w.DWORD), ('dwY', w.DWORD),
                ('dwXSize', w.DWORD), ('dwYSize', w.DWORD), ('dwXCountChars', w.DWORD),
                ('dwYCountChars', w.DWORD), ('dwFillAttribute', w.DWORD),
                ('dwFlags', w.DWORD), ('wShowWindow', w.WORD), ('cbReserved2', w.WORD),
                ('lpReserved2', ctypes.c_void_p), ('hStdInput', w.HANDLE),
                ('hStdOutput', w.HANDLE), ('hStdError', w.HANDLE)]


class ProcessInfo(ctypes.Structure):
    _fields_ = [('process', w.HANDLE), ('thread', w.HANDLE),
                ('pid', w.DWORD), ('tid', w.DWORD)]


class JobLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                ('flags', w.DWORD), ('min_working_set', ctypes.c_size_t),
                ('max_working_set', ctypes.c_size_t), ('active_processes', w.DWORD),
                ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]


class ExtendedJobLimits(ctypes.Structure):
    _fields_ = [('basic', JobLimits), ('io_counters', ctypes.c_uint64 * 6),
                ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]


def _api():
    user = ctypes.WinDLL('user32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = [
        (user, 'CreateDesktopW', [w.LPCWSTR, w.LPCWSTR, ctypes.c_void_p, w.DWORD, w.DWORD, ctypes.c_void_p], w.HANDLE),
        (user, 'CloseDesktop', [w.HANDLE], w.BOOL),
        (user, 'GetThreadDesktop', [w.DWORD], w.HANDLE),
        (user, 'GetUserObjectInformationW', [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD)], w.BOOL),
        (kernel, 'GetCurrentThreadId', [], w.DWORD),
        (kernel, 'CreateProcessW', [w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
                                  w.BOOL, w.DWORD, ctypes.c_void_p, w.LPCWSTR,
                                  ctypes.POINTER(StartupInfo), ctypes.POINTER(ProcessInfo)], w.BOOL),
        (kernel, 'CreateJobObjectW', [ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
        (kernel, 'SetInformationJobObject', [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
        (kernel, 'AssignProcessToJobObject', [w.HANDLE, w.HANDLE], w.BOOL),
        (kernel, 'ResumeThread', [w.HANDLE], w.DWORD),
        (kernel, 'WaitForSingleObject', [w.HANDLE, w.DWORD], w.DWORD),
        (kernel, 'GetExitCodeProcess', [w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
        (kernel, 'TerminateProcess', [w.HANDLE, w.UINT], w.BOOL),
        (kernel, 'CloseHandle', [w.HANDLE], w.BOOL),
    ]
    for dll, name, args, result in signatures:
        function = getattr(dll, name)
        function.argtypes, function.restype = args, result
    return user, kernel


def desktop_name():
    user, kernel = _api()
    handle = user.GetThreadDesktop(kernel.GetCurrentThreadId())
    name, length = ctypes.create_unicode_buffer(256), w.DWORD()
    if not user.GetUserObjectInformationW(handle, 2, name, ctypes.sizeof(name), ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    return name.value


def _read_capture(stream):
    stream.seek(0)
    return stream.read().decode('utf-8', errors='replace').replace('\r\n', '\n').replace('\r', '\n')


def run_gui(command, *, timeout, cwd=None, env=None):
    """Return captured output/exit code; terminate the whole test job on timeout.

    PIPELINE_GUI_TEST_MODE=interactive is an explicit opt-in for human review.
    All automated Windows callers default to desktop isolation.
    """
    environment = dict(os.environ if env is None else env)
    mode = environment.get('PIPELINE_GUI_TEST_MODE', 'isolated')
    if mode not in ('isolated', 'interactive'):
        raise ValueError('PIPELINE_GUI_TEST_MODE must be isolated or interactive')
    if sys.platform != 'win32' or mode == 'interactive':
        return subprocess.run(command, timeout=timeout, cwd=cwd, env=environment,
                              capture_output=True, text=True)
    import msvcrt

    user, kernel = _api()
    name = 'PipelineTests_' + uuid.uuid4().hex
    # Read/write objects, create windows/menus, hooks and enumeration; no switch right.
    desktop = user.CreateDesktopW(name, None, None, 0, 0x00CF, None)
    if not desktop:
        raise ctypes.WinError(ctypes.get_last_error())
    job, info = None, ProcessInfo()
    try:
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedJobLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        environment['PIPELINE_TEST_DESKTOP'] = name
        environment['PYTHONIOENCODING'] = 'utf-8'
        block = ctypes.create_unicode_buffer('\0'.join(f'{k}={v}' for k, v in
                                                       sorted(environment.items(), key=lambda item: item[0].upper())) + '\0')
        # Disk-backed capture cannot deadlock on a full pipe. Handles are made
        # inheritable only during creation and are closed even on launch failure.
        with open(os.devnull, 'rb') as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            handles = [msvcrt.get_osfhandle(f.fileno()) for f in (stdin, stdout, stderr)]
            startup = StartupInfo(cb=ctypes.sizeof(StartupInfo), lpDesktop=name,
                                  dwFlags=0x100 | 0x80,  # stdio + no busy cursor
                                  hStdInput=handles[0], hStdOutput=handles[1], hStdError=handles[2])
            try:
                for handle in handles:
                    os.set_handle_inheritable(handle, True)
                created = kernel.CreateProcessW(str(command[0]),
                    ctypes.create_unicode_buffer(subprocess.list2cmdline(list(map(str, command)))),
                    None, None, True, 0x08000000 | 0x400 | 0x4,  # no console, Unicode env, suspended
                    block, str(cwd) if cwd else None, ctypes.byref(startup), ctypes.byref(info))
                if not created:
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                for handle in handles:
                    os.set_handle_inheritable(handle, False)
            # Assign before any application code can create a child process.
            if not kernel.AssignProcessToJobObject(job, info.process):
                raise ctypes.WinError(ctypes.get_last_error())
            if kernel.ResumeThread(info.thread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + timeout
            while True:
                status = kernel.WaitForSingleObject(info.process, 50)
                if status == 0:
                    break
                if status != 258:
                    raise ctypes.WinError(ctypes.get_last_error())
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout,
                        output=_read_capture(stdout), stderr=_read_capture(stderr))
            code = w.DWORD()
            if not kernel.GetExitCodeProcess(info.process, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return subprocess.CompletedProcess(command, code.value,
                _read_capture(stdout), _read_capture(stderr))
    finally:
        if info.process:
            # Also covers failure assigning the still-suspended process to its job.
            kernel.TerminateProcess(info.process, 1)
        if job:
            kernel.CloseHandle(job)
        if info.process:
            kernel.WaitForSingleObject(info.process, 5000)
            kernel.CloseHandle(info.process)
        if info.thread:
            kernel.CloseHandle(info.thread)
        user.CloseDesktop(desktop)


def isolate_probe(timeout=45):
    """Guard standalone Python probes before creating their first Tk window."""
    if sys.platform != 'win32' or os.environ.get('PIPELINE_GUI_TEST_MODE') == 'interactive':
        return
    expected = os.environ.get('PIPELINE_TEST_DESKTOP')
    if expected and desktop_name() == expected:
        return
    result = run_gui([sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]], timeout=timeout)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('provide a command after --')
    result = run_gui(command, timeout=args.timeout)
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    raise SystemExit(result.returncode)
