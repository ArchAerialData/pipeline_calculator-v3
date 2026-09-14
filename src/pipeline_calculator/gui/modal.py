"""Shared modal surfaces with correctly composited rounded corners.

Tk widgets cannot alpha-composite their corners over sibling widgets. A uniform
backdrop gives every corner a real matching parent instead of painted black tips.
"""
import customtkinter as ctk
from pipeline_calculator.gui.scrolling import AutoScrollFrame

BACKDROP = '#191D23'
SURFACE = '#272D35'
TEXT = '#F1F4F8'
MUTED = '#B6C0CE'
OUTLINE = '#424C59'


class ModalSurface(ctk.CTkFrame):
    def __init__(self, root):
        super().__init__(root, fg_color=BACKDROP, corner_radius=0)
        self.card = ctk.CTkFrame(self, fg_color=SURFACE, bg_color=BACKDROP,
                                 corner_radius=18, border_width=1, border_color=OUTLINE)
        self.card.pack_propagate(False)
        self.card.place(relx=.5, rely=.5, anchor='center')
        self._preferred_size = None
        self.bind('<Configure>', self._fit_card, add='+')

    def show(self, preferred_size=None):
        self._preferred_size = preferred_size
        self.place(x=0, y=0, relwidth=1, relheight=1)
        self.lift()

    def _fit_card(self, event=None):
        if self._preferred_size is not None:
            scale = ctk.ScalingTracker.get_widget_scaling(self)
            width, height = self._preferred_size
            self.card.configure(width=min(width, self.winfo_width() / scale * .94),
                                height=min(height, self.winfo_height() / scale * .9))


class ModalBody(AutoScrollFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=SURFACE, scrollbar_button_color=OUTLINE,
                         scrollbar_button_hover_color='#637083')
