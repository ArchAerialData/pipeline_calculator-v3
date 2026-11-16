"""Shared card colors, wrapped typography, and deferred responsive layout."""
import customtkinter as ctk

from pipeline_calculator.gui.layout import WrappedLabel


BACKGROUND = '#20252C'
CARD = '#272D35'
OUTLINE = '#424C59'
TEXT = '#F1F4F8'
MUTED = '#B6C0CE'
GREEN = '#92DBAD'
BLUE = '#9CC8EB'
RED = '#FF8080'


def text_label(parent, text, *, size=16, color=MUTED, bold=False, **pack):
    label = WrappedLabel(parent, text=text, text_color=color, anchor='w', justify='left',
                         wrap_padding=2*pack.get('padx', 0),
                         font=ctk.CTkFont(size=size, weight='bold' if bold else 'normal'))
    label.pack(fill='x', **pack)
    return label


class DeferredLayoutFrame(ctk.CTkFrame):
    """Apply layout outside Tk's nested idle redraw callbacks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layout_id = None
        self._layout_host = self.winfo_toplevel()
        self.bind('<Configure>', self._queue_layout, add='+')

    def _queue_layout(self, event=None):
        if self._layout_id is None:
            self._layout_id = self._layout_host.after(20, self._layout)

    def _layout(self):
        self._layout_id = None
        self._arrange()

    def destroy(self):
        if self._layout_id is not None:
            self._layout_host.after_cancel(self._layout_id)
            self._layout_id = None
        super().destroy()

