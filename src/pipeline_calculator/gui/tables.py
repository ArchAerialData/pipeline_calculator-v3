"""Dark native tables with both scroll axes and DPI-scaled fonts/rows/columns."""
from tkinter import ttk
import customtkinter as ctk


def create_table(parent, columns, widths):
    frame = ctk.CTkFrame(parent)
    frame.pack(fill='both', expand=True, padx=8, pady=8)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    style = ttk.Style(parent)
    style.theme_use('clam')
    name = f'Pipeline{id(frame)}.Treeview'
    style.configure(name, background='#242424', foreground='#EEEEEE', fieldbackground='#242424',
                    borderwidth=0, lightcolor='#242424', darkcolor='#242424')
    style.configure(name+'.Heading', background='#343434', foreground='#EEEEEE', relief='flat')
    style.map(name, background=[('selected', '#1F538D')], foreground=[('selected', '#FFFFFF')])
    style.map(name+'.Heading', background=[('active', '#404040')])
    scroll_style = f'Pipeline{id(frame)}'
    for direction in ('Vertical', 'Horizontal'):
        style.configure(scroll_style+'.'+direction+'.TScrollbar', background='#454545',
                        troughcolor='#242424', arrowcolor='#EEEEEE', bordercolor='#242424',
                        lightcolor='#454545', darkcolor='#454545')
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=6, style=name)
    for column in columns:
        tree.heading(column, text=column)
    vsb = ttk.Scrollbar(frame, orient='vertical', command=tree.yview, style=scroll_style+'.Vertical.TScrollbar')
    hsb = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview, style=scroll_style+'.Horizontal.TScrollbar')
    tree.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
    tree.grid(row=0, column=0, sticky='nsew')
    vsb.grid(row=0, column=1, sticky='ns')
    hsb.grid(row=1, column=0, sticky='ew')
    last = [None]
    def scale_table(event=None):
        factor = ctk.ScalingTracker.get_widget_scaling(frame)
        if last[0] == factor:
            return
        last[0] = factor
        for direction in ('Vertical', 'Horizontal'):
            style.configure(scroll_style+'.'+direction+'.TScrollbar', arrowsize=round(14*factor))
        style.configure(name, rowheight=round(32*factor), font=('Arial', -round(13*factor)))
        style.configure(name+'.Heading', font=('Arial', -round(13*factor), 'bold'))
        for column, width in zip(columns, widths):
            tree.column(column, width=round(width*factor), minwidth=round(width*factor), stretch=True)
    frame.bind('<Configure>', scale_table, add='+')
    scale_table()
    return tree
