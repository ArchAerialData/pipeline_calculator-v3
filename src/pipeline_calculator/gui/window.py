"""Themed DnD window and monitor-aware sizing in logical/physical coordinates."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import sys
import tkinter as tk

import customtkinter as ctk
from tkinterdnd2 import TkinterDnD

BACKGROUND = '#181818'


def work_area(window):
    """Physical work area of the window's nearest monitor (excludes taskbar)."""
    if sys.platform == 'win32':
        try:
            class MonitorInfo(ctypes.Structure):
                _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', wintypes.RECT),
                            ('rcWork', wintypes.RECT), ('dwFlags', wintypes.DWORD)]
            user = ctypes.windll.user32
            user.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
            user.MonitorFromWindow.restype = wintypes.HANDLE
            user.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MonitorInfo)]
            user.GetMonitorInfoW.restype = wintypes.BOOL
            handle = user.MonitorFromWindow(window.winfo_id(), 2)
            info = MonitorInfo(cbSize=ctypes.sizeof(MonitorInfo))
            if user.GetMonitorInfoW(handle, ctypes.byref(info)):
                r = info.rcWork
                return r.left, r.top, r.right, r.bottom
        except (AttributeError, OSError, tk.TclError):
            pass
    return (window.winfo_vrootx(), window.winfo_vrooty(),
            window.winfo_vrootx()+window.winfo_screenwidth(),
            window.winfo_vrooty()+window.winfo_screenheight())


def fitted_geometry(area, scale, preferred=(1000, 720)):
    """Return logical client size and physical position, reserving native chrome."""
    left, top, right, bottom = area
    scale = max(.4, float(scale))
    margin = min(16, max(0, (right-left)//20), max(0, (bottom-top)//20))
    width = max(1, min(preferred[0], int((right-left-2*margin-16*scale)/scale)))
    height = max(1, min(preferred[1], int((bottom-top-2*margin-48*scale)/scale)))
    x = left + max(margin, int((right-left-width*scale-16*scale)/2))
    y = top + max(margin, int((bottom-top-height*scale-48*scale)/2))
    return width, height, x, y


def fit_window(window, preferred=(1000, 720), *, parent=None):
    window.update_idletasks()
    scale = ctk.ScalingTracker.get_window_scaling(window)
    width, height, x, y = fitted_geometry(work_area(parent or window), scale, preferred)
    window.minsize(min(440, width), min(340, height))
    # Tk geometry offsets with a negative sign mean distance from the far edge;
    # use +negative for absolute coordinates on left/above-primary monitors.
    window.geometry(f'{width}x{height}+{x}+{y}')


class AppWindow(ctk.CTk, TkinterDnD.DnDWrapper):
    """CTk supplies native dark chrome and per-monitor widget/window DPI tracking."""
    def __init__(self):
        ctk.set_appearance_mode('dark')
        ctk.set_default_color_theme('blue')
        super().__init__(fg_color=BACKGROUND)
        # CTk manages initial visibility itself. Calling withdraw() here latches
        # its pre-mainloop hidden-state flag; deiconify() does not clear it, so
        # the first mainloop() titlebar refresh hides an otherwise ready window.
        self.TkdndVersion = TkinterDnD._require(self)
        self._display = None
        self._fit_id = None
        self.bind('<Configure>', self._display_changed, add='+')

    def initialize_size(self):
        fit_window(self)
        self._display = (work_area(self), ctk.ScalingTracker.get_window_scaling(self))

    def _display_changed(self, event):
        if event.widget is not self or self.state() in ('withdrawn', 'iconic'):
            return
        display = (work_area(self), ctk.ScalingTracker.get_window_scaling(self))
        if display != self._display:
            self._display = display
            if self._fit_id is not None:
                self.after_cancel(self._fit_id)
            # CTk settles native DPI min/max constraints after 1 second.
            self._fit_id = self.after(1200, self._fit_display)

    def _fit_display(self):
        self._fit_id = None
        if self.state() != 'normal':
            return
        scale = ctk.ScalingTracker.get_window_scaling(self)
        fit_window(self, (round(self.winfo_width()/scale), round(self.winfo_height()/scale)))

    def destroy(self):
        if self._fit_id is not None:
            self.after_cancel(self._fit_id)
            self._fit_id = None
        # CTk also schedules titlebar/scaling callbacks without retaining their
        # handles. They must not outlive this interpreter's widgets.
        for callback in self.tk.splitlist(self.tk.call('after', 'info')):
            self.after_cancel(callback)
        super().destroy()
