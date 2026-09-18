"""Reusable content viewport with an overflow-only scrollbar."""
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from pipeline_calculator.gui.bindings import OwnedGlobalBindings


class AutoScrollFrame(ctk.CTkScrollableFrame):
    """Overflow-only scrolling; contain the CTk viewport adapter in one place."""
    def __init__(self, master, **kwargs):
        self._disposed = False
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
        # Tk can unmap an offscreen canvas window while its viewport stays
        # visible. Only the viewport's lifecycle should suspend maintenance.
        self._parent_canvas.bind('<Map>', self._mapped, add='+')
        self._parent_canvas.bind('<Unmap>', self._unmapped, add='+')

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
        if not isinstance(widget, tk.Misc) or not self._viewport_active():
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
        # Hidden content may have shrunk beyond the old scroll offset. Correct
        # that region before waiting for a content Map event that cannot arrive
        # while the canvas window is offscreen. Keep heavier layout deferred.
        if not self._viewport_active():
            return
        self._parent_canvas.configure(scrollregion=self._parent_canvas.bbox('all'))
        self._painted_fraction = None
        self._schedule_scrollbar()

    def _unmapped(self, event=None):
        self._cancel_scroll_callbacks()

    def _queue_scroll_position(self, first, last):
        if not self._viewport_active():
            return
        fraction = (float(first), float(last))
        self._scroll_fraction = fraction
        if (fraction != self._painted_fraction
                and self._scroll_position_id is None):
            self._scroll_position_id = self._callback_host.after(16, self._set_scroll_position)

    def _set_scroll_position(self):
        self._scroll_position_id = None
        if self._viewport_active() and self._scroll_fraction != self._painted_fraction:
            self._painted_fraction = self._scroll_fraction
            self._scrollbar.set(*self._scroll_fraction)

    def _schedule_scrollbar(self, event=None):
        if self._refresh_id is None and self._viewport_active():
            self._refresh_id = self._callback_host.after(25, self._refresh_scrollbar)

    def _refresh_scrollbar(self):
        self._refresh_id = None
        if not self._viewport_active() or self._parent_canvas.winfo_height() <= 1:
            return
        self._resize_viewport()
        # A viewport policy may enter Tk idle work while changing geometry.
        # Teardown during that work retires this refresh as well as its timers.
        if not self._viewport_active():
            return
        # On remap/content changes Tk clamps stale offsets to the current region.
        self._parent_canvas.configure(scrollregion=self._parent_canvas.bbox('all'))
        if not self._viewport_active():
            return
        overflow = self.winfo_reqheight() > self._parent_canvas.winfo_height() + 1
        if overflow and not self._scrollbar.winfo_manager():
            self._scrollbar.grid()
        elif not overflow and self._scrollbar.winfo_manager():
            self._scrollbar.grid_remove()
            self._parent_canvas.yview_moveto(0)
        self._queue_scroll_position(*self._parent_canvas.yview())

    def _resize_viewport(self):
        """Optional content-sized viewport policy (used by input settings)."""

    def _viewport_active(self):
        # CTk can leave the canvas wrapper alive after destroying its content.
        # The canvas must be visible, but live offscreen content still needs
        # maintenance to recover after a tab switch or a shorter scroll region.
        return (not self._disposed and self.winfo_exists()
                and self._parent_canvas.winfo_exists() and self._parent_canvas.winfo_viewable())

    def _cancel_scroll_callbacks(self):
        if self._scroll_position_id is not None:
            self._callback_host.after_cancel(self._scroll_position_id)
            self._scroll_position_id = None
        if self._refresh_id is not None:
            self._callback_host.after_cancel(self._refresh_id)
            self._refresh_id = None

    def destroy(self):
        self._disposed = True
        self._cancel_scroll_callbacks()
        self._global_bindings.close()
        super().destroy()
