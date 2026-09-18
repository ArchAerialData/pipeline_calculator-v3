"""Paged corridor table with native, keyboard-accessible row actions."""
from __future__ import annotations

from tkinter import ttk, messagebox
import customtkinter as ctk
from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.gui.layout import WrappedLabel
from pipeline_calculator.gui.sorting import HeaderSorter, sort_records
from pipeline_calculator.gui.tabs.summary_tab import number
from pipeline_calculator.gui.styles import corridor_button_style
from pipeline_calculator.gui.corridor_presentation import (
    corridor_is_omitted, corridor_unavailable_reason, MAP_OMISSION_NOTE,
)
from pipeline_calculator.export.corridor_metadata import MAP_NOTE, has_corridor_metadata


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
        self._map_reason = ''
        self._details_compact = False
        self._explanation_layout = None
        navigation = ctk.CTkFrame(self, fg_color='transparent')
        self.navigation = navigation
        navigation.pack(side='bottom', fill='x', padx=8, pady=(8, 4))
        self.previous = ctk.CTkButton(navigation, text='Previous', width=90, command=lambda: self.load_page(-1))
        self.previous.pack(side='left', padx=(0, 8))
        self.next_button = ctk.CTkButton(navigation, text='Next', width=90, command=lambda: self.load_page(1))
        self.next_button.pack(side='right', padx=(8, 0))
        self.page_label = WrappedLabel(navigation, text='', anchor='center')
        self.page_label.pack(fill='x', expand=True)
        self.map_explanation = WrappedLabel(self, text='', anchor='w', justify='left',
                                            text_color='#FFB993')
        self.tree = create_table(self, ('Pipeline Pair', 'Length (miles)', 'Avg Sep (m)', 'Action'),
                                 (440, 145, 130, 155), vertical_padding=0)
        self.button_style = corridor_button_style(self.tree, ctk.ScalingTracker.get_widget_scaling(self))
        self.map_details = ttk.Button(navigation, text='Map details', style=self.button_style,
                                      command=self._show_map_details, takefocus=True)
        self.map_details.bind('<Return>', lambda event: self._show_map_details())
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
        self.tree.bind('<<TreeviewSelect>>', self._show_selected_explanation, add='+')
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
            if not corridor_is_omitted(section):
                self.on_open_corridor(section, index)
            elif getattr(self, '_details_compact', False):
                self._select_for_explanation(item)
                self._show_map_details()

    def _double_click(self, event):
        if self.tree.identify_region(event.x, event.y) == 'cell':
            self._open(self.tree.identify_row(event.y))

    def _open_selected(self, event=None):
        if self.tree.selection():
            self._open(self.tree.selection()[0])
        return 'break'

    def _select_for_explanation(self, item):
        self.tree.selection_set(item)
        self.tree.focus(item)
        self.tree.focus_set()
        self._show_selected_explanation()

    def _show_selected_explanation(self, event=None):
        selection = self.tree.selection()
        record = self.item_map.get(selection[0]) if selection else None
        reason = corridor_unavailable_reason(record[0]) if record else ''
        self._map_reason = reason
        if reason:
            self.map_explanation.configure(text=f'Map unavailable: {reason} {MAP_OMISSION_NOTE}')
        self._layout_explanation()
        self._queue_position()

    def _layout_explanation(self):
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        self._details_compact = self.winfo_height() / scale < 210
        layout = (bool(self._map_reason), self._details_compact)
        if layout == self._explanation_layout:
            return
        self._explanation_layout = layout
        if self._map_reason and not self._details_compact:
            self._hide_map_details()
            self.map_explanation.pack(side='bottom', fill='x', padx=12, pady=(6, 2),
                                      before=self.tree.master)
        else:
            self.map_explanation.pack_forget()
            if self._map_reason:
                self.map_details.pack(side='right', padx=4, before=self.page_label)
            else:
                self._hide_map_details()

    def _hide_map_details(self):
        if self.focus_get() is self.map_details:
            self.tree.focus_set()
        self.map_details.pack_forget()

    def _show_map_details(self):
        if self._map_reason:
            messagebox.showinfo('Corridor map unavailable', f'{self._map_reason}\n\n{MAP_OMISSION_NOTE}', parent=self)
        return 'break'  # Prevent a second native Return activation on some platforms.

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
            omitted = corridor_is_omitted(section)
            button = ttk.Button(self.tree, text='Map unavailable' if omitted else 'View Corridor', style=self.button_style,
                                command=lambda item=item: self._open(item), takefocus=True)
            if omitted:
                button.configure(state='disabled')
                # The map stays disabled, but clicking its label selects the row
                # so the same explanation is reachable by pointer and keyboard.
                button.bind('<Button-1>', lambda event, item=item: self._select_for_explanation(item))
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
        self._show_selected_explanation()
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
        self._layout_explanation()
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        compact = self.winfo_height()/scale < 160
        if compact != self._compact:
            self._compact = compact
            self.navigation.pack(side='bottom', fill='x', padx=8, pady=0 if compact else (8, 4))
        if self._position_style != (scale, compact):
            self._position_style = (scale, compact)
            self.tree.set_row_height(30 if compact else 40)
            self.button_style = corridor_button_style(self.tree, scale)
            self.map_details.configure(style=self.button_style)
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
        if any(corridor_is_omitted(section) for section in sections):
            WrappedLabel(parent, text='Maps are approximate. Select an unavailable row for details.',
                         text_color='#FFB993', justify='left').pack(
                fill='x', padx=12, pady=(4, 6))
        elif has_corridor_metadata(current_results):
            WrappedLabel(parent, text=MAP_NOTE, text_color='#B6C0CE', justify='left').pack(
                fill='x', padx=12, pady=(4, 6))
        CorridorTable(parent, sections, on_open_corridor).pack(fill='both', expand=True, padx=4)
    else:
        state_failed = current_results.get('state_code') and current_results.get('adjusted_total_meters') is None
        combined_failed = current_results.get('overlap_analysis') is None and any(
            item.get('code') == 'overlap_analysis_failed'
            for item in (current_results.get('diagnostics') or []))
        failed = state_failed or combined_failed
        scope = 'State overlap' if current_results.get('state_code') else 'Overlap'
        WrappedLabel(parent, text=(f'{scope} analysis is unavailable. See Diagnostics for details.' if failed else
                                  'No bundled sections found with current parameters'),
                     font=('Arial', 14)).pack(fill='x', padx=16, pady=20)
