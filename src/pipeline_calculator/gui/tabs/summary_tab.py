"""Responsive mileage cards and a collapsed disclosure for supporting details."""
from __future__ import annotations

import math
import customtkinter as ctk
from pipeline_calculator.gui.disclosure import DisclosureButton
from pipeline_calculator.gui.layout import WrappedLabel
from pipeline_calculator.gui.scrolling import AutoScrollFrame
from pipeline_calculator.core.constants import SURVEY_MILE_METERS
from pipeline_calculator.gui.tables import create_table
from pipeline_calculator.export.corridor_metadata import corridor_detail_text, has_corridor_metadata

BACKGROUND = '#20252C'
CARD = '#272D35'
OUTLINE = '#424C59'
TEXT = '#F1F4F8'
MUTED = '#B6C0CE'
GREEN = '#92DBAD'
BLUE = '#9CC8EB'
RED = '#FF8080'


def add_status_notice(parent, current_results: dict) -> None:
    errors = [d.get('message', 'Analysis error') for d in (current_results.get('diagnostics') or [])
              if d.get('level') == 'error']
    if errors or current_results.get('analysis_complete') is False:
        messages = list(dict.fromkeys(errors))
        detail = '\n'.join(messages[:3])
        if len(messages) > 3:
            detail += f'\n{len(messages)-3} more issue(s); see Diagnostics or the exported workbook.'
        WrappedLabel(parent, text='Analysis incomplete. Totals cover loaded, valid geometry only.\n' + detail,
                     text_color='#FF8080', font=('Arial', 14, 'bold'), anchor='w', justify='left').pack(fill='x', pady=10)


def number(value, places=3):
    """Never substitute a plausible zero for a missing or non-finite estimate."""
    try:
        actual = float(value)
        if not math.isfinite(actual):
            return 'Unavailable'
        if 0 < actual < 10 ** -places:
            return '<' + f'{10 ** -places:.{places}f}'
        return f'{actual:,.{places}f}'
    except (TypeError, ValueError):
        return 'Unavailable'


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


class InlinePair(DeferredLayoutFrame):
    """Measured text columns: inline units, or bold facts that stack when needed."""
    def __init__(self, parent, label, value, *, metric=False, color=TEXT):
        super().__init__(parent, fg_color='transparent', width=1)
        self.metric = metric
        self._last_layout = None
        self.label_font = ctk.CTkFont(size=40 if metric else 18, weight='bold')
        self.value_font = ctk.CTkFont(size=18, weight='bold' if metric else 'normal')
        self.label = ctk.CTkLabel(self, text=label, width=1, anchor='sw' if metric else 'w', justify='left',
                                 text_color=color if metric else TEXT, font=self.label_font)
        self.value = ctk.CTkLabel(self, text=value, width=1, anchor='sw' if metric else 'w', justify='left',
                                 text_color=MUTED if metric else color, font=self.value_font)
        # Establish visible content before the first deferred width measurement.
        # Map/Configure timers can be delayed during DPI or scope transitions.
        self.grid_columnconfigure(1, weight=1)
        self.label.grid(row=0, column=0, sticky='w')
        self.value.grid(row=0, column=1, sticky='ew')

    def _arrange(self, event=None):
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        width = max(1, self.winfo_width() / scale)
        layout = (round(width, 2), scale, self.label.cget('text'), self.value.cget('text'))
        if layout == self._last_layout:
            return
        self._last_layout = layout
        gap = 12 if width >= 300 else 8
        if self.metric:
            size = 40 if width >= 340 else 32
            if self.label.cget('text') in ('Unavailable', 'Not applicable'):
                size = 28
            if self.label_font.cget('size') != size:
                self.label_font.configure(size=size)
            # Keep units beside the value; let the unit phrase wrap on phones.
            while size > 26 and self.label_font.measure(self.label.cget('text')) + gap + 72 > width:
                size -= 2
                self.label_font.configure(size=size)
            label_width = min(self.label_font.measure(self.label.cget('text')) + 2,
                              max(40, width - gap - 72))
            stacked = False
        else:
            size = 18 if width >= 340 else 16
            if self.label_font.cget('size') != size:
                self.label_font.configure(size=size)
            if self.value_font.cget('size') != size:
                self.value_font.configure(size=size)
            label_width = self.label_font.measure(self.label.cget('text')) + 2
            stacked = label_width + gap + min(140, self.value_font.measure(self.value.cget('text'))) > width
        self.grid_columnconfigure(0, weight=int(stacked))
        self.grid_columnconfigure(1, weight=int(not stacked))
        self.label.configure(width=1 if stacked else label_width,
                             wraplength=max(1, width if stacked else label_width))
        self.value.configure(wraplength=max(1, width if stacked else width-label_width-gap))
        # Align the last text baselines, including wrapped units on narrow screens.
        # Font descents differ, so bottom-aligning widget boxes alone is insufficient.
        baseline_pad = max(0, self.label_font.metrics('descent') - self.value_font.metrics('descent')) if self.metric else 0
        self.label.grid(row=0, column=0, sticky='sw' if self.metric else ('w' if not stacked else 'ew'))
        self.value.grid(row=1 if stacked else 0, column=0 if stacked else 1,
                        sticky='sew' if self.metric else 'ew',
                        padx=0 if stacked else (gap, 0),
                        pady=(0, baseline_pad) if self.metric else ((2, 0) if stacked else 0))


class EstimateCard(DeferredLayoutFrame):
    def __init__(self, master, title, value, *, adjusted=False):
        super().__init__(master, width=1, fg_color=CARD, corner_radius=18,
                         border_width=1, border_color=OUTLINE)
        self.content = ctk.CTkFrame(self, fg_color='transparent', width=1)
        self.content.pack(fill='both', expand=True, padx=28, pady=24)
        self.title = text_label(self.content, title, size=28, color=TEXT, bold=True, pady=(0, 20))
        self.metric = InlinePair(self.content, value, '(US survey miles)', metric=True,
                                 color=GREEN if adjusted else BLUE)
        self.metric.pack(fill='x', pady=(0, 24))
        self.value, self.unit = self.metric.label, self.metric.value
        ctk.CTkFrame(self.content, height=2, corner_radius=0, fg_color=OUTLINE).pack(fill='x', pady=(0, 14))
        self.facts = ctk.CTkFrame(self.content, fg_color='transparent', width=1)
        self.facts.pack(fill='x')
        self._last_layout = None

    def fact(self, text, color=MUTED):
        prefix, colon, value = text.partition(':')
        if colon:
            row = InlinePair(self.facts, prefix + ':', value.strip(), color=color)
            row.pack(fill='x', pady=6)
            return row
        return text_label(self.facts, text, size=18, color=color, pady=6)

    def _arrange(self, event=None):
        width = self.winfo_width() / ctk.ScalingTracker.get_widget_scaling(self)
        if round(width, 2) == self._last_layout:
            return
        self._last_layout = round(width, 2)
        self.content.pack(fill='both', expand=True, padx=28 if width >= 480 else 20, pady=24)
        self.title.cget('font').configure(size=28 if width >= 550 else (25 if width >= 380 else 22))


class SummaryView(AutoScrollFrame):
    def __init__(self, parent, results, *, on_select_state=None):
        super().__init__(parent, fg_color=BACKGROUND, corner_radius=12,
                         scrollbar_button_color=OUTLINE, scrollbar_button_hover_color='#637083')
        self.results = results
        self.geography = results.get('_geography') or results.get('geography')
        self.state_view = bool(results.get('state_code'))
        self.on_select_state = on_select_state
        self.expanded = False
        self._arrangement = None
        self.inner = ctk.CTkFrame(self, fg_color='transparent')
        self.inner.pack(fill='x', padx=20, pady=(16, 24))
        text_label(self.inner, 'Analysis Summary', size=26, color=TEXT, bold=True, pady=(0, 4))
        text_label(self.inner, 'Original mileage and the estimate after qualifying overlaps are removed.', pady=(0, 20))
        add_status_notice(self.inner, results)
        if self.geography and self.geography.get('status') != 'complete':
            text_label(self.inner, 'State breakdown ' + str(self.geography.get('status', 'incomplete')) +
                       ('. Select Combined to review all state results and their status.' if self.state_view else
                        '. Combined analysis remains available. Review state statuses and diagnostics.'),
                       color='#FFB993', pady=(0, 12))
        self.cards = ctk.CTkFrame(self.inner, fg_color='transparent')
        self.cards.pack(fill='x')
        pipelines = results.get('pipelines') or []
        diagnostics = results.get('diagnostics') or []
        errors = sum(d.get('level') == 'error' for d in diagnostics)
        warnings = sum(d.get('level') == 'warning' for d in diagnostics)
        incomplete = results.get('analysis_complete') is False or errors > 0
        status = 'Incomplete' if incomplete else ('Complete' if results.get('analysis_complete') is True else 'Status not recorded')
        status += f' · {errors} error' + ('' if errors == 1 else 's')
        if warnings:
            status += f' · {warnings} warning' + ('' if warnings == 1 else 's')
        original_title = ('Mileage assigned to ' + results['state_name']
                          if self.state_view else 'Total Pipeline Mileage')
        self.original = EstimateCard(self.cards, original_title, number(results.get('total_miles')))
        self.original.fact(f'Pipelines Analyzed: {len(pipelines):,}', TEXT)
        status_color = '#FFB993' if incomplete else ('#E5C783' if warnings else
                       (GREEN if results.get('analysis_complete') is True else MUTED))
        self.original.fact(f'Run Status: {status}', status_color)
        if self.state_view and results.get('shared_allocation_meters', 0) > 0:
            allocation = number(results['shared_allocation_meters'] / SURVEY_MILE_METERS)
            self.original.fact(f'Includes {allocation} mi of shared-border allocation.')
        overlap = results.get('overlap_analysis')
        has_estimate = overlap is not None and number(overlap.get('effective_total_miles')) != 'Unavailable'
        self.adjusted = EstimateCard(self.cards, 'Overlap Adjusted Mileage',
            number(overlap.get('effective_total_miles')) if has_estimate else
            (number(results['adjusted_total_meters'] / SURVEY_MILE_METERS)
             if self.state_view and results.get('adjusted_total_meters') is not None else
             ('Not applicable' if len(pipelines) < 2 and not self.state_view else 'Unavailable')),
            adjusted=True)
        if has_estimate:
            savings, percent = number(overlap.get('savings_miles')), number(overlap.get('savings_percentage'), 1)
            self.adjusted.fact(f'Mileage Removed: {savings} mi ({percent}%)' if savings != 'Unavailable' and percent != 'Unavailable'
                               else 'Mileage Removed: not recorded', RED)
            self.adjusted.fact(f'Bundled Sections: {len(overlap.get("bundled_sections") or []):,}')
        elif self.state_view and results.get('adjusted_total_meters') is not None:
            self.adjusted.fact('Mileage Removed: 0.000 mi (0.0%)', RED)
            self.adjusted.fact('No qualifying interior overlap.')
        else:
            self.adjusted.unit.configure(text='No adjusted estimate')
            self.adjusted.fact('Overlap analysis requires at least two pipelines.' if len(pipelines) < 2 and not self.state_view
                               else 'Adjusted mileage was not produced for this run.')
            self.adjusted.fact('See the run status and Diagnostics for details.' if incomplete else 'Original mileage is shown in the first card.')
        if self.state_view and results.get('shared_allocation_meters', 0) > 0:
            text_label(self.inner, 'Shared-border overlap: Not calculated. Shared allocations receive no state overlap discount.',
                       color=MUTED, pady=(12, 0))
        if self.geography and not self.state_view:
            self._create_state_comparison()
        self.disclosure = ctk.CTkFrame(self.inner, fg_color=CARD, corner_radius=14, border_width=1, border_color=OUTLINE)
        self.disclosure.pack(fill='x', pady=(20, 0))
        self.toggle = DisclosureButton(self.disclosure, command=self.toggle_details, text='Additional details & run settings  ▸',
                                anchor='w', justify='left', bg=CARD, fg=TEXT, activebackground='#34404E',
                                activeforeground=TEXT, relief='flat', borderwidth=0, highlightthickness=1,
                                highlightbackground=CARD, highlightcolor=BLUE, takefocus=True, cursor='hand2', padx=8, pady=8)
        self.toggle.pack(fill='x', padx=12, pady=12)
        self.toggle.bind('<FocusIn>', self._reveal_toggle, add='+')
        self.details = ctk.CTkFrame(self.disclosure, fg_color='transparent')
        self._details_created = False
        self._arrange_id = None
        self.bind('<Configure>', self._queue_arrange, add='+')
        self.bind('<Map>', self._queue_arrange, add='+')
        self.bind('<Unmap>', self._cancel_arrange, add='+')
        # Start with a valid stacked layout, then adapt to the mapped width.
        # A new scope must never depend on a timer just to display its cards.
        self.cards.grid_columnconfigure(0, weight=1)
        self.original.grid(row=0, column=0, sticky='nsew', pady=(0, 16))
        self.adjusted.grid(row=1, column=0, sticky='nsew')

    def _queue_arrange(self, event=None):
        if self.winfo_viewable() and self._arrange_id is None:
            self._arrange_id = self._callback_host.after(20, self._apply_arrange)

    def _apply_arrange(self):
        self._arrange_id = None
        if self.winfo_viewable():
            self._arrange()

    def _cancel_arrange(self, event=None):
        if self._arrange_id is not None:
            self._callback_host.after_cancel(self._arrange_id)
            self._arrange_id = None

    def destroy(self):
        self._cancel_arrange()
        super().destroy()

    def _create_details(self, diagnostics, overlap):
        text_label(self.details, 'Run settings', size=16, color=TEXT, bold=True, pady=(0, 8))
        params = self.results.get('analysis_parameters') or {}
        for title, key, unit in [('Detection range', 'detection_range', 'm'),
                                 ('Minimum parallel length', 'min_parallel_length', 'm'),
                                 ('Segment length', 'segment_length', 'm'),
                                 ('Angular tolerance', 'angular_tolerance', '°')]:
            value = params.get(key)
            text_label(self.details, f'{title}: {value} {unit}' if value is not None else f'{title}: Not recorded', pady=2)
        if has_corridor_metadata(self.results):
            text_label(self.details, 'Corridor maps', size=16, color=TEXT, bold=True, pady=(20, 8))
            text_label(self.details, corridor_detail_text(self.results), pady=2)
        text_label(self.details, 'Run details', size=16, color=TEXT, bold=True, pady=(20, 8))
        counts = {level: sum(d.get('level') == level for d in diagnostics) for level in ('error', 'warning', 'info')}
        text_label(self.details, f'Diagnostics: {counts["error"]} errors · {counts["warning"]} warnings · {counts["info"]} informational', pady=2)
        text_label(self.details, f'Placemarks: {len(self.results.get("placemarks") or []):,}', pady=2)
        if self.results.get('parsed_kml_files') is not None:
            text_label(self.details, f'KML documents read: {len(self.results["parsed_kml_files"]):,}', pady=2)
        sections = (overlap or {}).get('bundled_sections') or []
        if sections:
            total = sum(float(section.get('bundled_length_miles', 0)) for section in sections)
            text_label(self.details, f'Pairwise bundled length: {number(total)} US survey miles', pady=2)
        text_label(self.details, 'How to read these estimates', size=16, color=TEXT, bold=True, pady=(20, 8))
        text_label(self.details, 'Pipeline count reflects loaded pipeline records. Bundled sections are pairwise sections, '
                   'so their count does not represent unique geographic areas. Pairwise bundled length can exceed the mileage removed.', pady=(0, 8))
        text_label(self.details, 'Mileage removed is the reduction from original to adjusted mileage. Corridors are approximate visual guides; '
                   'KML descriptions identify rectangle fallbacks. See Diagnostics for individual notices.')
        if self.geography:
            text_label(self.details, 'State boundary analysis', size=16, color=TEXT, bold=True, pady=(20, 8))
            provenance = self.geography.get('boundary_source') or {}
            for label, key in [('Boundary source', 'name'), ('Boundary vintage', 'vintage'),
                                ('Source URL', 'source_url'), ('Resource checksum', 'resource_sha256')]:
                if provenance.get(key):
                    text_label(self.details, f'{label}: {provenance[key]}', pady=2)
            text_label(self.details, 'State original mileage includes interior mileage plus an equal share of verified shared borders. '
                       'State overlaps qualify independently, so state adjusted mileage and savings may differ from the Combined result.', pady=(4, 8))
            text_label(self.details, 'State pipeline counts represent original source pipelines, including pipelines with shared allocations. '
                       'Map line mileage represents interior mileage; shared lines appear once in the Combined map.', pady=(0, 8))
            reconciliation = self.geography.get('reconciliation') or {}
            for label, key in [('Original', 'source_meters'), ('Assigned to states', 'attributed_state_meters'),
                                ('Outside coverage', 'outside_meters'), ('Unresolved', 'unresolved_meters')]:
                if reconciliation.get(key) is not None:
                    text_label(self.details, f'{label}: {number(reconciliation[key] / SURVEY_MILE_METERS)} mi', pady=2)
            text_label(self.details, 'Mileage reconciliation: ' + ('Passed' if reconciliation.get('passed') else 'Incomplete'), pady=2)
            text_label(self.details, 'Boundary precision is limited by the source data. These results do not establish surveyed ownership.', pady=(8, 0))

    def _create_state_comparison(self):
        states = sorted(self.geography.get('states') or [], key=lambda row: row['state_name'])
        crossings = self.geography.get('crossing_count')
        heading = f'States represented: {len(states)}'
        if crossings is not None:
            heading += f' · Border crossings: {crossings}'
        text_label(self.inner, heading, size=18, color=TEXT, bold=True, pady=(20, 6))
        reconciliation = self.geography.get('reconciliation') or {}
        shared = reconciliation.get('shared_meters', 0)
        if shared > 0:
            text_label(self.inner, f'{number(shared / SURVEY_MILE_METERS)} mi follows shared borders and is divided equally '
                       'among adjoining states. Shared allocations receive no state overlap discount.', pady=(0, 6))
        exceptions = [f'{label}: {number(reconciliation[key] / SURVEY_MILE_METERS)} mi'
                      for label, key in [('Outside coverage', 'outside_meters'), ('Unresolved', 'unresolved_meters')]
                      if reconciliation.get(key, 0) > 0]
        if exceptions:
            text_label(self.inner, ' · '.join(exceptions) + '. This mileage is retained separately from state totals.',
                       color='#FFB993', pady=(0, 6))
        if reconciliation.get('passed') is False:
            text_label(self.inner, 'State mileage does not reconcile. The state breakdown is incomplete.',
                       color=RED, pady=(0, 6))
        text_label(self.inner, 'Select a state to inspect its mileage and overlaps. Separate paths may occupy different states without crossing a border.',
                   pady=(0, 6))
        if not states:
            text_label(self.inner, 'No state mileage is available for this input.', pady=(0, 8))
            return
        table = create_table(self.inner, ('State', 'Original (mi)', 'Adjusted (mi)', 'Removed (mi)', 'Status'),
                             (160, 130, 130, 130, 150), vertical_padding=0, compact=True)
        table.configure(height=min(6, max(1, len(states))))
        self.state_table = table
        self.state_rows = {}
        for state in states:
            adjusted = state.get('adjusted_total_meters')
            savings = state.get('interior_savings_meters')
            status = state.get('status') or ('Complete' if state.get('analysis_complete') else 'Incomplete')
            item = table.insert('', 'end', values=(state['state_name'], number(state.get('total_miles')),
                number(adjusted / SURVEY_MILE_METERS if adjusted is not None else None),
                number(savings / SURVEY_MILE_METERS if savings is not None else None), str(status).capitalize()))
            self.state_rows[item] = state['state_name']
        def choose(event=None):
            row = (table.identify_row(event.y) if event is not None and hasattr(event, 'y') and event.type.name == 'ButtonRelease'
                   else next(iter(table.selection()), None))
            if row in self.state_rows and self.on_select_state is not None:
                self.on_select_state(self.state_rows[row])
            return 'break'
        table.bind('<ButtonRelease-1>', choose)
        table.bind('<Return>', choose)
        table.bind('<FocusIn>', lambda event: self._reveal_widget(table), add='+')

    def _reveal_toggle(self, event=None):
        """Tab navigation must reveal the disclosure even below a short viewport."""
        self._reveal_widget(self.toggle)

    def _reveal_widget(self, widget):
        canvas = self._parent_canvas
        top = widget.winfo_rooty() - self.winfo_rooty()
        bottom = top + widget.winfo_height()
        start, height = canvas.canvasy(0), canvas.winfo_height()
        if top < start:
            canvas.yview_moveto(top / max(1, self.winfo_height()))
        elif bottom > start + height:
            canvas.yview_moveto((bottom - height) / max(1, self.winfo_height()))

    def toggle_details(self, event=None):
        self.expanded = not self.expanded
        self.toggle.configure(text='Additional details & run settings  ' + ('▾' if self.expanded else '▸'))
        if self.expanded:
            if not self._details_created:
                self._create_details(self.results.get('diagnostics') or [], self.results.get('overlap_analysis'))
                self._details_created = True
            self.details.pack(fill='x', padx=24, pady=(0, 24))
        else:
            self.details.pack_forget()
        self._schedule_scrollbar()
        return 'break'  # Do not also invoke the native Button class's Return binding.

    def _arrange(self, event=None):
        scale = ctk.ScalingTracker.get_widget_scaling(self)
        width = self.winfo_width() / scale
        margin = max(16, (width - 1440) / 2)
        content_width = max(1, width - 2*margin)
        columns = 2 if content_width >= 900 else 1
        arrangement = (round(margin), columns, round(scale, 2), round(content_width))
        if arrangement == self._arrangement:
            return
        self._arrangement = arrangement
        self.inner.pack(fill='x', padx=margin, pady=(16, 24))
        for column in range(2):
            self.cards.grid_columnconfigure(column, weight=int(column < columns),
                                            uniform='estimates' if column < columns else '')
        for index, card in enumerate((self.original, self.adjusted)):
            card.grid(row=index//columns, column=index%columns, sticky='nsew',
                      padx=(0, 12) if columns == 2 and index == 0 else ((12, 0) if columns == 2 else 0),
                      pady=(0, 16) if columns == 1 and index == 0 else 0)
        self.toggle.configure(font=('Segoe UI', -round(16*scale), 'bold'),
                              wraplength=max(100, round((content_width-64)*scale)),
                              padx=round(8*scale), pady=round(8*scale))


def create(parent, current_results: dict, *, on_select_state=None) -> None:
    SummaryView(parent, current_results, on_select_state=on_select_state).pack(fill='both', expand=True, padx=16, pady=16)
