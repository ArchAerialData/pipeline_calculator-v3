"""Content-sized settings viewport with an overflow-only scrollbar.

CTkScrollableFrame does not expose viewport/scrollbar controls publicly. Keep the
small adapter to its 5.2.x internals here rather than spreading it across pages.
"""
import math
import customtkinter as ctk
from pipeline_calculator.gui.scrolling import AutoScrollFrame


class SettingsPanel(AutoScrollFrame):
    def __init__(self, master):
        self.maximum_height = 240
        super().__init__(master, fg_color='#202020', height=240, corner_radius=0)

    def _schedule_refresh(self, event=None):
        self._schedule_scrollbar(event)

    def _resize_viewport(self):
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        height = min(self.maximum_height, max(1, math.ceil(self.winfo_reqheight() / scale)))
        if self.cget('height') != height:
            self.configure(height=height)
