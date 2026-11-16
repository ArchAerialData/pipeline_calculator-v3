"""Main-thread layer and CRS decisions; source discovery stays on the worker."""
from __future__ import annotations

import time
import tkinter as tk
import re

import customtkinter as ctk

from pipeline_calculator.gui.layout import ActionBar, WrappedLabel
from pipeline_calculator.gui.modal import MUTED, TEXT
from pipeline_calculator.gui.repair_ui import RepairPanel, _keyboard_button


class ImportPanel(RepairPanel):
    """Build bounded batches of selectable layers in the shared scrollable modal."""

    def __init__(self, root, source, *, on_apply, on_cancel, on_choose_another, draft=None,
                 missing_crs=()):
        self.source = source
        self.on_apply = on_apply
        self.layers = list(source.layers)
        self.missing_crs = set(missing_crs)
        self.selection_vars = {}
        self.selection_widgets = {}
        self.crs_vars = {}
        self.rows = []
        self.load_id = None
        self.draft = draft or {'layer_ids': source.selected_layers,
                               'crs_overrides': getattr(source, 'crs_overrides', {})}
        self.selected_ids = set(self.draft.get('layer_ids', source.selected_layers))
        super().__init__(root, title='Choose GIS layers', filename=source.display_name,
            message='Select the layers to analyze. Separate parts stay separate. '
                    'For a missing coordinate system, enter the EPSG code or WKT supplied with the dataset.',
            actions=[('Cancel', on_cancel), ('Choose another file', on_choose_another)],
            primary_action=('Analyze selected layers', self.apply), on_cancel=on_cancel)
        self.error_label = WrappedLabel(self.body, text='', text_color='#FFB993', anchor='w', justify='left')
        self.error_label.pack(fill='x', pady=(0, 8))
        self.bulk_actions = ActionBar(self.body, [('Select all', lambda: self.select_all(True)),
                                                 ('Clear selection', lambda: self.select_all(False))])
        self.bulk_actions.pack(fill='x', pady=(0, 8))
        for button in self.bulk_actions.buttons:
            _keyboard_button(button)
            self.focus_controls.append(button)
        self.layer_body = ctk.CTkFrame(self.body, fg_color='transparent')
        self.layer_body.pack(fill='x')
        self.footer.primary_button.configure(state='disabled')
        self._load_rows()

    def _load_rows(self):
        self.load_id = None
        if self.closed:
            return
        deadline = time.perf_counter() + .012
        for _ in range(8):
            if len(self.rows) >= len(self.layers):
                break
            layer = self.layers[len(self.rows)]
            layer_id = layer['id']
            supported = bool(layer.get('supported'))
            row = ctk.CTkFrame(self.layer_body, fg_color='#303944')
            row.pack(fill='x', pady=(0, 8))
            self.rows.append(row)
            variable = tk.BooleanVar(master=self.root, value=supported and layer_id in self.selected_ids)
            self.selection_vars[layer_id] = variable
            check = ctk.CTkCheckBox(row, text='', width=28, variable=variable,
                state='normal' if supported else 'disabled')
            check.pack(side='left', anchor='n', padx=(10, 2), pady=12)
            self.selection_widgets[layer_id] = check
            content = ctk.CTkFrame(row, fg_color='transparent')
            content.pack(side='left', fill='both', expand=True, padx=(2, 12), pady=10)
            WrappedLabel(content, text=str(layer.get('name') or layer_id), text_color=TEXT,
                font=ctk.CTkFont(size=14, weight='bold'), anchor='w', justify='left').pack(fill='x')
            count = layer.get('feature_count')
            count_text = f'{count:,} features' if isinstance(count, int) and count >= 0 else 'Feature count unavailable'
            crs = layer.get('crs')
            needs_crs = not crs or layer_id in self.missing_crs
            description = f"{layer.get('geometry_type') or 'Unknown geometry'} · {count_text}\nDataset: {layer.get('dataset') or source_name(self.source)}"
            description += '\nCoordinate system: ' + ('Missing or unrecognized' if needs_crs else crs_label(layer))
            if not supported:
                description += '\nNot selectable: ' + str(layer.get('reason') or 'Unsupported layer')
            WrappedLabel(content, text=description, text_color=MUTED, anchor='w', justify='left').pack(fill='x', pady=(2, 4))
            if supported:
                tk.Misc.configure(check, takefocus=1)
                tk.Misc.bind(check, '<space>', lambda event, widget=check: (widget.toggle(), 'break')[1])
                self.focus_controls.append(check)
            if supported and needs_crs:
                WrappedLabel(content, text='Coordinate system supplied by the data provider',
                    text_color=TEXT, anchor='w', justify='left').pack(fill='x', pady=(4, 2))
                crs_var = tk.StringVar(master=self.root,
                    value=self.draft.get('crs_overrides', {}).get(layer_id, ''))
                self.crs_vars[layer_id] = crs_var
                entry = ctk.CTkEntry(content, textvariable=crs_var, placeholder_text='For example, EPSG:4269')
                entry.pack(fill='x', pady=(0, 4))
                self.focus_controls.append(entry._entry)
            if time.perf_counter() >= deadline:
                break
        if len(self.rows) < len(self.layers):
            self.load_id = self.root.after(10, self._load_rows)
        else:
            self.footer.primary_button.configure(state='normal')
        self._queue_resize()

    def select_all(self, selected):
        self.selected_ids = {layer['id'] for layer in self.layers if selected and layer.get('supported')}
        for key, variable in self.selection_vars.items():
            variable.set(key in self.selected_ids)

    def apply(self):
        if self.closed or len(self.rows) != len(self.layers):
            return
        selected = [layer['id'] for layer in self.layers
                    if layer.get('supported') and self.selection_vars[layer['id']].get()]
        if not selected:
            self.error_label.configure(text='Select at least one supported layer.')
            return
        overrides = {key: variable.get().strip() for key, variable in self.crs_vars.items() if key in selected}
        if any(not value for value in overrides.values()):
            self.error_label.configure(text='Provide a coordinate system for each selected layer marked Missing, or deselect it.')
            return
        self.on_apply({'layer_ids': selected, 'crs_overrides': overrides})

    def close(self):
        if self.load_id is not None:
            self.root.after_cancel(self.load_id)
            self.load_id = None
        super().close()


def source_name(source):
    return getattr(source, 'display_name', 'Selected source')


def crs_label(layer):
    """Show a bounded name without resolving CRS definitions on the Tk thread."""
    value = layer.get('crs_label') or layer.get('crs')
    if not value:
        return 'Missing'
    text = str(value)
    wkt_name = re.match(r'\s*\w+\s*\[\s*"([^"]+)"', text)
    if wkt_name:
        text = wkt_name.group(1)
    return text if len(text) <= 180 else text[:177] + '…'
