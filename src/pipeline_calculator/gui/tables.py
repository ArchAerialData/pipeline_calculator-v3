"""Dark native tables with both scroll axes and DPI-scaled fonts/rows/columns."""
from tkinter import ttk, font as tkfont
import customtkinter as ctk
from pipeline_calculator.gui.styles import table_styles


class ContentTable(ttk.Treeview):
    """Content-sized columns for small summary tables, with DPI-aware padding."""

    def __init__(self, parent, columns, **kwargs):
        super().__init__(parent, columns=columns, **kwargs)
        self._columns = tuple(columns)
        self._content_widths = dict.fromkeys(columns, 0)
        self._fit_id = None
        self._factor = None

    def set_scale(self, factor):
        if factor == self._factor:
            return
        self._factor = factor
        self._body_font = tkfont.Font(self, family='Arial', size=-round(13 * factor))
        self._heading_font = tkfont.Font(self, family='Arial', size=-round(13 * factor), weight='bold')
        self._content_widths = dict.fromkeys(self._columns, 0)
        for item in self.get_children():
            self._measure_values(self.item(item, 'values'))
        self.fit_columns()

    def _measure_values(self, values):
        for column, value in zip(self._columns, values):
            self._content_widths[column] = max(
                self._content_widths[column], self._body_font.measure(str(value)))

    def insert(self, parent, index, iid=None, **kwargs):
        item = super().insert(parent, index, iid=iid, **kwargs)
        self._measure_values(kwargs.get('values', ()))
        self.fit_columns()
        return item

    def fit_columns(self):
        if self._fit_id is None:
            self._fit_id = self.after_idle(self._apply_widths)

    def _apply_widths(self):
        self._fit_id = None
        # Include native cell/heading insets in addition to our styled padding.
        padding = 2 * (round(12 * self._factor) + 5)
        for column in self._columns:
            heading = self.heading(column)
            image = self.tk.splitlist(heading['image'])
            icon_width = int(self.tk.call('image', 'width', image[0])) if image else 0
            width = max(self._heading_font.measure(heading['text']) + icon_width + padding,
                        self._content_widths[column] + padding)
            if self.column(column, 'width') != width or self.column(column, 'minwidth') != width:
                self.column(column, width=width, minwidth=width, stretch=False)
        # Tk caches its requested size when column widths change. Refresh the
        # geometry request so the compact container also resizes after a DPI change.
        self.configure(height=self.cget('height'))

    def destroy(self):
        if self._fit_id is not None:
            self.after_cancel(self._fit_id)
        self._fit_id = None
        super().destroy()


def create_table(parent, columns, widths, *, vertical_padding=8, compact=False):
    if compact:
        host = ctk.CTkFrame(parent, fg_color='transparent')
        host.pack(fill='both', expand=True, padx=8, pady=vertical_padding)
        frame = ctk.CTkFrame(host, corner_radius=0)
        # Limit the natural width to the viewport without stretching columns.
        frame.pack(anchor='w', fill='y', expand=True)
    else:
        frame = ctk.CTkFrame(parent)
        frame.pack(fill='both', expand=True, padx=8, pady=vertical_padding)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    factor = ctk.ScalingTracker.get_widget_scaling(frame)
    scroll_style = table_styles(frame, factor, 32, compact=compact)
    name = scroll_style + '.Treeview'
    tree_class = ContentTable if compact else ttk.Treeview
    tree = tree_class(frame, columns=columns, show='headings', height=6, style=name)
    for column in columns:
        tree.heading(column, text=column, anchor='w')
        tree.column(column, anchor='w')
    vsb = ttk.Scrollbar(frame, orient='vertical', command=tree.yview, style=scroll_style+'.Vertical.TScrollbar')
    hsb = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview, style=scroll_style+'.Horizontal.TScrollbar')
    def scroll_changed(scrollbar, first, last):
        scrollbar.set(first, last)
        if compact and float(first) <= 0 and float(last) >= 1:
            scrollbar.grid_remove()
        else:
            scrollbar.grid()
    tree.configure(xscrollcommand=lambda first, last: scroll_changed(hsb, first, last),
                   yscrollcommand=lambda first, last: scroll_changed(vsb, first, last))
    tree.grid(row=0, column=0, sticky='nsew')
    vsb.grid(row=0, column=1, sticky='ns')
    hsb.grid(row=1, column=0, sticky='ew')
    last = [None]
    row_height = [32]
    def scale_table(event=None):
        factor = ctk.ScalingTracker.get_widget_scaling(frame)
        key = (factor, row_height[0])
        if last[0] == key:
            return
        last[0] = key
        style_name = table_styles(frame, factor, row_height[0], compact=compact)
        tree.configure(style=style_name + '.Treeview')
        vsb.configure(style=style_name + '.Vertical.TScrollbar')
        hsb.configure(style=style_name + '.Horizontal.TScrollbar')
        if compact:
            tree.set_scale(factor)
        else:
            for column, width in zip(columns, widths):
                tree.column(column, width=round(width*factor), minwidth=round(width*factor), stretch=True)
    def set_row_height(height):
        if row_height[0] != height:
            row_height[0] = height
            scale_table()
    tree.set_row_height = set_row_height
    frame.bind('<Configure>', scale_table, add='+')
    scale_table()
    return tree
