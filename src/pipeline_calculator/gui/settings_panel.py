"""Content-sized settings viewport with an overflow-only scrollbar.

CTkScrollableFrame does not expose viewport/scrollbar controls publicly. Keep the
small adapter to its 5.2.x internals here rather than spreading it across pages.
"""
import math
import customtkinter as ctk


class SettingsPanel(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master, fg_color='#202020', height=240, corner_radius=0)
        self._refresh_id = None
        self.maximum_height = 240
        self.bind('<Configure>', self._schedule_refresh, add='+')
        self._parent_canvas.bind('<Configure>', self._schedule_refresh, add='+')

    def _schedule_refresh(self, event=None):
        if self._refresh_id is None:
            # Avoid recursive CTk layout updates while its scrollbar is drawing.
            self._refresh_id = self.after(25, self._refresh)

    def _refresh(self):
        self._refresh_id = None
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        height = min(self.maximum_height, max(1, math.ceil(self.winfo_reqheight() / scale)))
        if self.cget('height') != height:
            self.configure(height=height)
        overflow = self.winfo_reqheight() > self._parent_canvas.winfo_height() + 1
        shown = bool(self._scrollbar.winfo_manager())
        if overflow and not shown:
            self._scrollbar.grid()
        elif not overflow and shown:
            self._scrollbar.grid_remove()
            self._parent_canvas.yview_moveto(0)

    def destroy(self):
        if self._refresh_id is not None:
            self.after_cancel(self._refresh_id)
            self._refresh_id = None
        super().destroy()
