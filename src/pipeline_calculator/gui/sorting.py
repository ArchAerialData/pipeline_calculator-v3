"""Shared stable sorting of source values and single-column header indicators."""
import math
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageTk


def sort_records(records, key, *, numeric=False, descending=False):
    """Sort full precision values; missing values stay last in either direction."""
    present, missing = [], []
    for record in records:
        value = key(record)
        if value is None or str(value).strip() == '':
            missing.append(record)
            continue
        if numeric:
            try:
                value = float(str(value).replace(',', ''))
                if not math.isfinite(value):
                    missing.append(record)
                    continue
                value = (0, value)
            except (ValueError, TypeError):
                # IDs may mix numbers and textual identifiers.
                value = (1, str(value).casefold())
        else:
            value = str(value).casefold()
        present.append((value, record))
    return [record for _, record in sorted(present, key=lambda pair: pair[0], reverse=descending)] + missing


class HeaderSorter:
    """A click selects ascending order; repeating it reverses that column."""
    def __init__(self, tree, columns, on_sort):
        self.tree, self.on_sort = tree, on_sort
        self.column = None
        self.descending = False
        self.titles = {column: tree.heading(column, 'text') for column in columns}
        self._scale = None
        self.images = {}
        for column in columns:
            tree.heading(column, anchor=tree.column(column, 'anchor'), command=lambda column=column: self.sort(column))
        tree.bind('<Configure>', self._indicators, add='+')
        self._indicators()

    def _indicators(self, event=None):
        factor = ctk.ScalingTracker.get_widget_scaling(self.tree.master)
        if self._scale != factor:
            self._scale = factor
            size = max(16, round(20*factor))
            images = {}
            for state in ('neutral', 'ascending', 'descending'):
                icon = Image.new('RGBA', (80, 80))
                draw = ImageDraw.Draw(icon)
                if state == 'neutral':
                    draw.polygon([(40, 10), (16, 32), (64, 32)], fill='#A6B3C2')
                    draw.polygon([(16, 48), (64, 48), (40, 70)], fill='#A6B3C2')
                else:
                    points = [(40, 24), (12, 54), (68, 54)] if state == 'ascending' else [(12, 26), (68, 26), (40, 56)]
                    draw.polygon(points, fill='#A9D9FF')
                images[state] = ImageTk.PhotoImage(icon.resize((size, size), Image.Resampling.LANCZOS), master=self.tree)
            self.images = images
        for column in self.titles:
            state = ('descending' if self.descending else 'ascending') if column == self.column else 'neutral'
            self.tree.heading(column, image=self.images[state])

    def sort(self, column):
        self.descending = not self.descending if self.column == column else False
        self.column = column
        self._indicators()
        self.on_sort(column, self.descending)
