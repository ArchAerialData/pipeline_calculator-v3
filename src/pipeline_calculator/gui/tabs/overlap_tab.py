"""Paged corridor table with native, keyboard-accessible row actions."""
from __future__ import annotations

from tkinter import ttk
import customtkinter as ctk
from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.layout import WrappedLabel
from pipeline_calculator.gui.sorting import HeaderSorter, sort_records
from pipeline_calculator.gui.tabs.summary_tab import number
from pipeline_calculator.gui.styles import corridor_button_style


class CorridorTable(ctk.CTkFrame):
    PAGE_SIZE = 20

    def __init__(self, parent, sections, on_open_corridor):
        super().__init__(parent, fg_color='transparent')
        self.sections, self.on_open_corridor = sections, on_open_corridor
        self.ordered_sections = list(enumerate(sections, start=1))
        self.page = 0
        self.item_map = {}
        self.row_buttons = {}
        self._position_id = None
        self._callback_host = self.winfo_toplevel()
        self._compact = None
        navigation = ctk.CTkFrame(self, fg_color='transparent')
        self.navigation = navigation
        navigation.pack(side='bottom', fill='x', padx=8, pady=(8, 4))
        self.previous = ctk.CTkButton(navigation, text='Previous', width=90, command=lambda: self.load_page(-1))
        self.previous.pack(side='left', padx=(0, 8))
        self.next_button = ctk.CTkButton(navigation, text='Next', width=90, command=lambda: self.load_page(1))
        self.next_button.pack(side='right', padx=(8, 0))
        self.page_label = WrappedLabel(navigation, text='', anchor='center')
        self.page_label.pack(fill='x', expand=True)
        self.tree = create_table(self, ('Pipeline Pair', 'Length (miles)', 'Avg Sep (m)', 'Action'),
                                 (440, 145, 130, 155), vertical_padding=0)
        self.button_style = corridor_button_style(self.tree, ctk.ScalingTracker.get_widget_scaling(self))
        self._position_style = None
        for column in self.tree['columns']:
            self.tree.heading(column, anchor='w')
        # Scroll notifications include wheel, keyboard and scrollbar navigation.
        for axis in ('x', 'y'):
            original = self.tree.cget(axis+'scrollcommand')
            def scrolled(first, last, original=original):
                self.tree.tk.call(*self.tree.tk.splitlist(original), first, last)
                self._queue_position()
            self.tree.configure(**{axis+'scrollcommand': scrolled})
        self.tree.bind('<Configure>', self._queue_position, add='+')
        self.tree.bind('<Map>', self._queue_position, add='+')
        self.tree.bind('<Unmap>', self._cancel_position, add='+')
        self.tree.bind('<ButtonRelease-1>', self._queue_position, add='+')
        self.tree.bind('<Double-1>', self._double_click)
        self.tree.bind('<Return>', self._open_selected)
        self.bind('<Configure>', self._queue_position, add='+')
        self.sorter = HeaderSorter(self.tree, ('Pipeline Pair', 'Length (miles)', 'Avg Sep (m)'), self._sort)
        self.load_page()

    def _sort(self, column, descending):
        keys = {'Pipeline Pair': lambda row: f"{row[1].get('pipeline_1')} + {row[1].get('pipeline_2')}",
                'Length (miles)': lambda row: row[1].get('bundled_length_miles'),
                'Avg Sep (m)': lambda row: row[1].get('average_separation')}
        self.ordered_sections = sort_records(self.ordered_sections, keys[column],
                                            numeric=column != 'Pipeline Pair', descending=descending)
        self.page = 0
        self.load_page()

    def _open(self, item):
        if item in self.item_map:
            section, index = self.item_map[item]
            if section.get('clipped_polygons') != []:
                self.on_open_corridor(section, index)

    def _double_click(self, event):
        if self.tree.identify_region(event.x, event.y) == 'cell':
            self._open(self.tree.identify_row(event.y))

    def _open_selected(self, event=None):
        if self.tree.selection():
            self._open(self.tree.selection()[0])
        return 'break'

    def load_page(self, delta=0):
        last = (len(self.sections)-1)//self.PAGE_SIZE
        self.page = max(0, min(last, self.page+delta))
        for button in self.row_buttons.values():
            button.destroy()
        self.row_buttons.clear()
        self.item_map.clear()
        self.tree.delete(*self.tree.get_children())
        first = self.page*self.PAGE_SIZE
        end = min(first+self.PAGE_SIZE, len(self.sections))
        for position in range(first, end):
            index, section = self.ordered_sections[position]
            item = self.tree.insert('', 'end', values=(
                f"{section.get('pipeline_1')} + {section.get('pipeline_2')}",
                number(section.get('bundled_length_miles', 0)),
                f"{section.get('average_separation', 0):.1f}", ''))
            self.item_map[item] = (section, index)
            omitted = section.get('clipped_polygons') == []
            button = ttk.Button(self.tree, text='Map unavailable' if omitted else 'View Corridor', style=self.button_style,
                                command=lambda item=item: self._open(item), takefocus=True)
            if omitted:
                button.configure(state='disabled')
            button.bind('<Return>', lambda event, item=item: self._activate_button(item))
            self.row_buttons[item] = button
        self.tree.yview_moveto(0)
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
        self.previous.configure(state='normal' if self.page else 'disabled')
        self.next_button.configure(state='normal' if self.page < last else 'disabled')
        self.page_label.configure(text=f'{first+1}–{end} of {len(self.sections)}')
        self._queue_position()

    def _activate_button(self, item):
        self._open(item)
        return 'break'

    def _queue_position(self, event=None):
        if self._position_id is None and self.winfo_viewable():
            self._position_id = self._callback_host.after(20, self._position_buttons)

    def _position_buttons(self):
        self._position_id = None
        if not self.winfo_viewable():
            return
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        compact = self.winfo_height()/scale < 160
        if compact != self._compact:
            self._compact = compact
            self.navigation.pack(side='bottom', fill='x', padx=8, pady=0 if compact else (8, 4))
        if self._position_style != (scale, compact):
            self._position_style = (scale, compact)
            self.tree.set_row_height(30 if compact else 40)
            self.button_style = corridor_button_style(self.tree, scale)
            for button in self.row_buttons.values():
                button.configure(style=self.button_style)
            for column in ('Length (miles)', 'Avg Sep (m)', 'Action'):
                self.tree.column(column, stretch=False)
        pad = round(4*scale)
        for item, button in self.row_buttons.items():
            box = self.tree.bbox(item, 'Action')
            if box:
                x, y, width, height = box
                # Never paint over the heading, scrollbars or the footer when a
                # row/cell is partially outside the viewport.
                if x >= 0 and x+width <= self.tree.winfo_width() and y+height <= self.tree.winfo_height():
                    button.place(x=x+pad, y=y+pad, width=min(width-2*pad, round(140*scale)), height=height-2*pad)
                    continue
            button.place_forget()

    def _cancel_position(self, event=None):
        if self._position_id is not None:
            self._callback_host.after_cancel(self._position_id)
            self._position_id = None

    def destroy(self):
        self._cancel_position()
        super().destroy()


def create(parent, current_results: dict, *, on_open_corridor) -> None:
    if current_results.get('state_code') and current_results.get('shared_allocation_meters', 0) > 0:
        WrappedLabel(parent, text='Shared-border overlap: Not calculated. Only interior geometry is analyzed here.',
                     text_color='#B6C0CE').pack(fill='x', padx=12, pady=8)
    sections = (current_results.get('overlap_analysis') or {}).get('bundled_sections') or []
    if sections:
        CorridorTable(parent, sections, on_open_corridor).pack(fill='both', expand=True, padx=4)
    else:
        failed = current_results.get('state_code') and current_results.get('adjusted_total_meters') is None
        WrappedLabel(parent, text=('State overlap analysis is unavailable. See Diagnostics for details.' if failed else
                                  'No bundled sections found with current parameters'),
                     font=('Arial', 14)).pack(pady=20)
