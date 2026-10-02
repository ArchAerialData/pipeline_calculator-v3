"""Combined point inventory and responsive cards for its containing folders."""
from __future__ import annotations

from collections import Counter
import time

import customtkinter as ctk

from pipeline_calculator.core.placemarks import summarize_placemarks
from pipeline_calculator.gui.cards import (
    BACKGROUND, CARD, OUTLINE, TEXT, BLUE, text_label, DeferredLayoutFrame,
)
from pipeline_calculator.gui.scrolling import AutoScrollFrame


class PointCountCard(ctk.CTkFrame):
    def __init__(self, master, title, count, *, caption='', combined=False):
        super().__init__(master, width=1, fg_color=CARD, corner_radius=16,
                         border_width=1, border_color=OUTLINE)
        self.title = text_label(self, title, size=24 if combined else 20,
                                color=TEXT, bold=True, padx=20, pady=(18, 8))
        self.value = text_label(self, 'Not recorded' if count is None else f'{count:,}',
                                size=38 if combined else 32, color=BLUE, bold=True, padx=20)
        self.unit = text_label(self, 'point pins', size=14, padx=20, pady=(0, 12))
        self.caption = text_label(self, caption, size=13, padx=20, pady=(0, 18))


class FolderCardGrid(DeferredLayoutFrame):
    """Use the available width, with consistent spacing and no fixed card height."""
    def __init__(self, master):
        super().__init__(master, fg_color='transparent', width=1)
        self.cards = []
        self.columns = 1
        self.grid_columnconfigure(0, weight=1, uniform='folders')

    def add(self, card):
        index = len(self.cards)
        self.cards.append(card)
        self._place(card, index)
        self._queue_layout()

    def _place(self, card, index):
        card.grid(row=index // self.columns, column=index % self.columns,
                  sticky='nsew', padx=6, pady=6)

    def _arrange(self):
        width = self.winfo_width() / ctk.ScalingTracker.get_widget_scaling(self)
        columns = max(1, min(4, len(self.cards), int(width // 280)))
        if columns == self.columns:
            return
        self.columns = columns
        for column in range(4):
            self.grid_columnconfigure(column, weight=int(column < columns),
                                      uniform='folders' if column < columns else '', minsize=0)
        for index, card in enumerate(self.cards):
            self._place(card, index)


class PlacemarksView(AutoScrollFrame):
    """A scrollable overview shared by the modern and legacy desktop entrypoints."""
    def __init__(self, parent, results):
        self._load_id = None
        super().__init__(parent, fg_color=BACKGROUND, corner_radius=12,
                         scrollbar_button_color=OUTLINE, scrollbar_button_hover_color='#637083')
        # A label-only overview still needs a keyboard path to content below
        # the fold. Scope these bindings to this focusable viewport.
        self._parent_canvas.configure(takefocus=True, highlightthickness=1,
                                      highlightbackground=BACKGROUND, highlightcolor=BLUE)
        for sequence, direction, units in [('<Up>', -1, 'units'), ('<Down>', 1, 'units'),
                                            ('<Prior>', -1, 'pages'), ('<Next>', 1, 'pages')]:
            self._parent_canvas.bind(sequence, lambda event, d=direction, u=units: self._scroll_key(d, u))
        self._parent_canvas.bind('<Home>', lambda event: self._scroll_key(0))
        self._parent_canvas.bind('<End>', lambda event: self._scroll_key(1))
        self.inventory = summarize_placemarks(results.get('placemarks'))
        self.groups = self.inventory['groups']
        self._show_sources = len({group['source_kml'] for group in self.groups}) > 1
        self._same_paths = Counter((group['source_kml'], tuple(group['folder_path']), group['name'])
                                   for group in self.groups)
        self._path_occurrences = Counter()
        self.inner = ctk.CTkFrame(self, fg_color='transparent', width=1)
        self.inner.pack(fill='x', padx=14, pady=(16, 22))
        text_label(self.inner, 'Placemarks overview', size=26, color=TEXT, bold=True, pady=(0, 4))
        text_label(self.inner, 'Point pins from the loaded source files. Full records are available in the XLSX export.',
                   pady=(0, 16))
        total = self.inventory['total']
        if total is None:
            caption = 'This saved result does not include a point inventory. Reimport the source file to record it.'
        elif not total:
            caption = 'No point pins were found in the loaded geometry.'
        else:
            count = self.inventory['folder_count']
            caption = (f'{count:,} subfolder' + ('s' if count != 1 else '') + ' with point pins.'
                       if count else 'Folder details appear below.')
        self.total_card = PointCountCard(self.inner, 'Combined total', total, caption=caption, combined=True)
        self.total_card.pack(fill='x', pady=(0, 16))
        self.folder_grid = FolderCardGrid(self.inner)
        self.folder_cards = self.folder_grid.cards
        if self.groups:
            text_label(self.inner, 'By subfolder', size=20, color=TEXT, bold=True, pady=(0, 4))
            text_label(self.inner, 'Each pin is counted once in its containing folder. Nested folders have separate cards.',
                       size=14, pady=(0, 8))
            self.folder_grid.pack(fill='x')
        self.loading_label = text_label(self.inner, '', size=13, pady=(8, 0))
        self.loading_label.pack_forget()
        # Start a bounded batch immediately. Subsequent batches yield and stop
        # work while the tab is hidden or being replaced.
        self._add_cards(limit=8)
        self._parent_canvas.bind('<Map>', self._resume_cards, add='+')

    def _caption(self, group):
        path = group['folder_path']
        parts = [' / '.join(path[:-1])] if len(path) > 1 else []
        if group['name'] == 'Folder not recorded' and not path:
            parts.append('Reimport the source file to recover folder names.')
        elif not path:
            parts.append('Pins outside any subfolder.')
        elif len(path) == 1:
            parts.append('Top-level folder')
        if self._show_sources and group['source_kml']:
            parts.append('Source: ' + group['source_kml'])
        key = (group['source_kml'], tuple(path), group['name'])
        self._path_occurrences[key] += 1
        if self._same_paths[key] > 1:
            parts.append(f'Same-name folder {self._path_occurrences[key]} of {self._same_paths[key]}')
        return '\n'.join(parts)

    def _scroll_key(self, direction, units=None):
        if units is None:
            self._parent_canvas.yview_moveto(direction)
        else:
            self._parent_canvas.yview_scroll(direction, units)
        return 'break'

    def _add_cards(self, *, limit):
        deadline = time.perf_counter() + .012
        for _ in range(limit):
            if len(self.folder_cards) >= len(self.groups):
                break
            group = self.groups[len(self.folder_cards)]
            card = PointCountCard(self.folder_grid, group['name'], group['count'], caption=self._caption(group))
            self.folder_grid.add(card)
            if time.perf_counter() >= deadline:
                break
        if len(self.folder_cards) < len(self.groups):
            self.loading_label.configure(text=f'Loading subfolders: {len(self.folder_cards):,} of {len(self.groups):,}')
            self.loading_label.pack(fill='x', pady=(8, 0))
        else:
            self.loading_label.pack_forget()

    def _resume_cards(self, event=None):
        if (self._viewport_active() and self._load_id is None
                and len(self.folder_cards) < len(self.groups)):
            self._load_id = self._callback_host.after(10, self._load_cards)

    def _load_cards(self):
        self._load_id = None
        if self._viewport_active():
            self._add_cards(limit=8)
            self._resume_cards()

    def _cancel_card_loading(self):
        if self._load_id is not None:
            self._callback_host.after_cancel(self._load_id)
            self._load_id = None

    def _unmapped(self, event=None):
        self._cancel_card_loading()
        super()._unmapped(event)

    def destroy(self):
        self._cancel_card_loading()
        super().destroy()


def create(parent, current_results: dict) -> None:
    PlacemarksView(parent, current_results).pack(fill='both', expand=True, padx=8, pady=8)
