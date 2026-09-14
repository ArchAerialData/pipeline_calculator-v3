"""Reusable content viewport with an overflow-only scrollbar."""
import customtkinter as ctk


class AutoScrollFrame(ctk.CTkScrollableFrame):
    """Overflow-only scrolling; contain the CTk viewport adapter in one place."""
    def __init__(self, master, **kwargs):
        kwargs.setdefault('corner_radius', 0)
        kwargs.setdefault('height', 160)
        super().__init__(master, **kwargs)
        self._refresh_id = None
        self._callback_host = self.winfo_toplevel()
        self._scroll_position_id = None
        self._scroll_fraction = None
        # CTkScrollbar.set() calls update_idletasks(). Calling it directly from
        # a canvas layout notification can recursively enter another resize.
        self._parent_canvas.configure(yscrollcommand=self._queue_scroll_position)
        self.bind('<Configure>', self._schedule_scrollbar, add='+')
        self._parent_canvas.bind('<Configure>', self._schedule_scrollbar, add='+')

    def _queue_scroll_position(self, first, last):
        fraction = (float(first), float(last))
        if fraction != self._scroll_fraction:
            self._scroll_fraction = fraction
            if self._scroll_position_id is None:
                self._scroll_position_id = self._callback_host.after(16, self._set_scroll_position)

    def _set_scroll_position(self):
        self._scroll_position_id = None
        self._scrollbar.set(*self._scroll_fraction)

    def _schedule_scrollbar(self, event=None):
        if self._refresh_id is None:
            self._refresh_id = self._callback_host.after(25, self._refresh_scrollbar)

    def _refresh_scrollbar(self):
        self._refresh_id = None
        overflow = self.winfo_reqheight() > self._parent_canvas.winfo_height() + 1
        if overflow and not self._scrollbar.winfo_manager():
            self._scrollbar.grid()
        elif not overflow and self._scrollbar.winfo_manager():
            self._scrollbar.grid_remove()
            self._parent_canvas.yview_moveto(0)

    def destroy(self):
        if self._scroll_position_id is not None:
            self._callback_host.after_cancel(self._scroll_position_id)
            self._scroll_position_id = None
        if self._refresh_id is not None:
            self._callback_host.after_cancel(self._refresh_id)
            self._refresh_id = None
        super().destroy()
