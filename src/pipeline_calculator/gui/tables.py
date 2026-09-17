"""Dark native tables with both scroll axes and DPI-scaled fonts/rows/columns."""
from tkinter import ttk
import customtkinter as ctk
from pipeline_calculator.gui.styles import table_styles


def create_table(parent, columns, widths, *, vertical_padding=8):
    frame = ctk.CTkFrame(parent)
    frame.pack(fill='both', expand=True, padx=8, pady=vertical_padding)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    factor = ctk.ScalingTracker.get_widget_scaling(frame)
    scroll_style = table_styles(frame, factor, 32)
    name = scroll_style + '.Treeview'
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=6, style=name)
    for column in columns:
        tree.heading(column, text=column, anchor='w')
        tree.column(column, anchor='w')
    vsb = ttk.Scrollbar(frame, orient='vertical', command=tree.yview, style=scroll_style+'.Vertical.TScrollbar')
    hsb = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview, style=scroll_style+'.Horizontal.TScrollbar')
    tree.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
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
        style_name = table_styles(frame, factor, row_height[0])
        tree.configure(style=style_name + '.Treeview')
        vsb.configure(style=style_name + '.Vertical.TScrollbar')
        hsb.configure(style=style_name + '.Horizontal.TScrollbar')
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
