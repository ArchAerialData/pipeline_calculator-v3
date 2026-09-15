"""Reusable content viewport with an overflow-only scrollbar."""
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from pipeline_calculator.gui.bindings import OwnedGlobalBindings


class AutoScrollFrame(ctk.CTkScrollableFrame):
    """Overflow-only scrolling; contain the CTk viewport adapter in one place."""
    def __init__(self, master, **kwargs):
        self._global_bindings = OwnedGlobalBindings()
        kwargs.setdefault('corner_radius', 0)
        kwargs.setdefault('height', 160)
        super().__init__(master, **kwargs)
        self._refresh_id = None
        self._callback_host = self.winfo_toplevel()
        self._scroll_position_id = None
        self._scroll_fraction = None
        self._painted_fraction = None
        # CTkScrollbar.set() calls update_idletasks(). Calling it directly from
        # a canvas layout notification can recursively enter another resize.
        self._parent_canvas.configure(yscrollcommand=self._queue_scroll_position)
        self.bind('<Configure>', self._schedule_scrollbar, add='+')
        self._parent_canvas.bind('<Configure>', self._schedule_scrollbar, add='+')
        self.bind('<Map>', self._mapped, add='+')
        self.bind('<Unmap>', self._unmapped, add='+')

    def bind_all(self, sequence=None, func=None, add=None):
        # Read Shift from each wheel event, so changing windows while a key is
        # held cannot leave scrolling stuck horizontally. No global key hooks.
        if sequence and 'Shift_' in sequence:
            return None
        return self._global_bindings.add(self, sequence, func, add)

    def _mouse_wheel_all(self, event):
        # Tk may supply a string for widgets owned by native popups. Nested
        # tables/text/listboxes consume their own wheel, including at the edge.
        widget = event.widget
        if not isinstance(widget, tk.Misc) or not self.winfo_viewable():
            return
        while widget is not None and widget is not self._parent_canvas:
            if isinstance(widget, (ttk.Treeview, tk.Text, tk.Listbox)):
                return
            if isinstance(widget, AutoScrollFrame) and widget is not self:
                return
            widget = widget.master
        if widget is self._parent_canvas:
            self._shift_pressed = bool(event.state & 1)
            super()._mouse_wheel_all(event)

    def _mapped(self, event=None):
        self._painted_fraction = None
        self._schedule_scrollbar()

    def _unmapped(self, event=None):
        self._cancel_scroll_callbacks()

    def _queue_scroll_position(self, first, last):
        fraction = (float(first), float(last))
        self._scroll_fraction = fraction
        if (fraction != self._painted_fraction and self.winfo_viewable()
                and self._scroll_position_id is None):
            self._scroll_position_id = self._callback_host.after(16, self._set_scroll_position)

    def _set_scroll_position(self):
        self._scroll_position_id = None
        if self.winfo_viewable() and self._scroll_fraction != self._painted_fraction:
            self._painted_fraction = self._scroll_fraction
            self._scrollbar.set(*self._scroll_fraction)

    def _schedule_scrollbar(self, event=None):
        if self._refresh_id is None and self.winfo_viewable():
            self._refresh_id = self._callback_host.after(25, self._refresh_scrollbar)

    def _refresh_scrollbar(self):
        self._refresh_id = None
        if not self.winfo_viewable() or self._parent_canvas.winfo_height() <= 1:
            return
        self._resize_viewport()
        # On remap/content changes Tk clamps stale offsets to the current region.
        self._parent_canvas.configure(scrollregion=self._parent_canvas.bbox('all'))
        overflow = self.winfo_reqheight() > self._parent_canvas.winfo_height() + 1
        if overflow and not self._scrollbar.winfo_manager():
            self._scrollbar.grid()
        elif not overflow and self._scrollbar.winfo_manager():
            self._scrollbar.grid_remove()
            self._parent_canvas.yview_moveto(0)
        self._queue_scroll_position(*self._parent_canvas.yview())

    def _resize_viewport(self):
        """Optional content-sized viewport policy (used by input settings)."""

    def _cancel_scroll_callbacks(self):
        if self._scroll_position_id is not None:
            self._callback_host.after_cancel(self._scroll_position_id)
            self._scroll_position_id = None
        if self._refresh_id is not None:
            self._callback_host.after_cancel(self._refresh_id)
            self._refresh_id = None

    def destroy(self):
        self._cancel_scroll_callbacks()
        self._global_bindings.close()
        super().destroy()
