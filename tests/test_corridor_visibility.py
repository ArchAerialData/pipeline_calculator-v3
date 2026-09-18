"""Unavailable geography visuals cannot be launched by pointer or keyboard routes."""
from types import SimpleNamespace

import pytest

from pipeline_calculator.gui.actions.corridor_launch import launch_corridor, corridor_is_omitted
from pipeline_calculator.gui.corridor_presentation import corridor_unavailable_reason
from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable


@pytest.mark.parametrize('decision', [
    {'visualization_status': 'omitted'}, {'clipped_polygons': []},
    {'visualization_schema_version': 1, 'visualization_status': 'ready'},
    {'visualization_schema_version': 1, 'visualization_status': 'ready', 'visualization_polygons': None},
])
def test_omitted_corridor_cannot_launch_even_with_a_valid_original_fallback(decision):
    section = dict(decision, bbox={'min_lon': -100, 'max_lon': -99, 'min_lat': 30, 'max_lat': 31})
    launches = []
    table = SimpleNamespace(item_map={'row': (section, 1)},
                            on_open_corridor=lambda *args: launches.append(args))
    CorridorTable._open(table, 'row')
    assert launches == []
    with pytest.raises(ValueError, match='map is unavailable'):
        launch_corridor(None, section, 1)


@pytest.mark.native_gui
def test_omitted_corridor_row_is_labelled_and_disabled():
    import customtkinter as ctk
    root = ctk.CTk()
    sections = [{'pipeline_1': 'A', 'pipeline_2': 'B', 'visualization_status': status}
                for status in ('omitted', 'ready')]
    sections[1]['visualization_polygons'] = [
        {'outer': [[0, 0], [1, 0], [0, 1], [0, 0]], 'holes': []}]
    launches = []
    table = CorridorTable(root, sections, lambda *args: launches.append(args))
    try:
        table.pack(fill='both', expand=True)
        root.update()
        first, second = table.tree.get_children()
        assert table.row_buttons[first].cget('text') == 'Map unavailable'
        assert table.row_buttons[first].instate(['disabled'])
        table.tree.selection_set(first)
        table._open_selected()
        table.row_buttons[first].invoke()
        assert launches == []
        assert table.row_buttons[second].cget('text') == 'View Corridor'
        assert table.row_buttons[second].instate(['!disabled'])
        table.tree.selection_set(second)
        table._open_selected()
        assert launches == [(sections[1], 2)]
    finally:
        root.destroy()


@pytest.mark.parametrize('code,context,expected', [
    ('corridor_buffer_limit', {}, 'drawing complexity or resource limit'),
    ('state_corridor_omitted', {'reason_code': 'corridor_buffer_limit'}, 'drawing complexity or resource limit'),
    ('corridor_projection_unavailable', {}, 'geographic accuracy checks'),
    ('corridor_geometry_invalid', {}, 'safely constructed'),
    ('corridor_coverage_failed', {}, 'all qualifying pipeline paths'),
    ('state_corridor_omitted', {}, 'clipped and verified within this state'),
    ('state_corridor_unavailable', {}, 'could not be prepared'),
    ('unknown', {'error': 'budget geometry containment'}, 'could not be generated'),
])
def test_omission_reason_uses_structured_codes_not_arbitrary_error_text(code, context, expected):
    section = {'visualization_status': 'omitted', 'visualization_polygons': [],
               'diagnostics': [{'code': code, 'context': context}]}
    assert expected in corridor_unavailable_reason(section)
    with pytest.raises(ValueError, match=expected):
        launch_corridor(None, section, 1)


def test_missing_unsupported_and_valid_canonical_geometry_have_distinct_explanations():
    missing = {'visualization_schema_version': 1, 'visualization_status': 'ready'}
    assert 'missing or incomplete' in corridor_unavailable_reason(missing)
    unsupported = dict(missing, visualization_schema_version=2)
    assert 'format is not supported' in corridor_unavailable_reason(unsupported)
    # Clipped multipart geometry with holes takes precedence over the uncut field.
    polygons = [
        {'outer': [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
         'holes': [[[.5, .5], [.5, 1], [1, 1], [1, .5], [.5, .5]]]},
        {'outer': [[3, 0], [4, 0], [4, 1], [3, 1], [3, 0]], 'holes': []},
    ]
    valid = dict(missing, clipped_polygons=polygons, visualization_polygons=[])
    assert not corridor_is_omitted(valid)
    assert corridor_unavailable_reason(valid) == ''


@pytest.mark.native_gui
@pytest.mark.parametrize('scale,size', [(1.0, '1100x500'), (1.5, '1100x500'), (1.5, '640x360')])
def test_selected_row_explanation_is_accessible_and_fits_without_enabling_map(scale, size):
    import customtkinter as ctk
    import time

    ctk.set_widget_scaling(scale)
    root = ctk.CTk()
    root.geometry(size)
    unavailable = {'pipeline_1': 'A', 'pipeline_2': 'B', 'visualization_status': 'omitted',
                   'diagnostics': [{'code': 'corridor_buffer_limit'}]}
    available = {'pipeline_1': 'C', 'pipeline_2': 'D'}  # Supported legacy approximation.
    state = dict(unavailable, pipeline_1='Z',
                 diagnostics=[{'code': 'state_corridor_omitted'}])
    opened = []
    table = CorridorTable(root, [unavailable, available, *[available]*18, state],
                          lambda *args: opened.append(args))

    def settle():
        until = time.monotonic() + .2
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)

    try:
        table.pack(fill='both', expand=True)
        settle()
        first, second, *_ = table.tree.get_children()
        assert table.row_buttons[first].instate(['disabled'])
        assert table.map_explanation.winfo_ismapped()
        assert 'drawing complexity' in table.map_explanation.cget('text')
        assert 'completed' not in table.map_explanation.cget('text')
        assert table.map_explanation.winfo_height() >= table.map_explanation.winfo_reqheight()
        assert table.tree.winfo_height() > 45 * scale
        table.tree.selection_set(second)
        settle()
        assert not table.map_explanation.winfo_ismapped()
        # Clicking the disabled map control reveals the reason, never launches.
        table.row_buttons[first].event_generate('<Button-1>')
        settle()
        assert table.tree.selection() == (first,)
        assert table.map_explanation.winfo_ismapped()
        table.tree.event_generate('<Return>')
        table.row_buttons[first].invoke()
        assert opened == []
        table.load_page(1)
        settle()
        assert 'within this state' in table.map_explanation.cget('text')
        table._sort('Pipeline Pair', True)
        settle()
        assert 'within this state' in table.map_explanation.cget('text')
        table.tree.selection_set(table.tree.get_children()[1])
        settle()
        assert not table.map_explanation.winfo_ismapped()
    finally:
        root.destroy()
        ctk.set_widget_scaling(1.0)


@pytest.mark.native_gui
@pytest.mark.parametrize('scale', [1.0, 1.5])
def test_full_results_page_keeps_omitted_map_rows_usable_in_short_viewport(monkeypatch, scale):
    import customtkinter as ctk
    from pipeline_calculator.gui.layout import ResultPages
    from pipeline_calculator.gui.pages.results_page import show
    from pipeline_calculator.gui.tabs import overlap_tab
    from test_repair_ui import activate_for_keyboard
    from test_results_header import assert_inside
    from test_state_breakdown_ui import sample_results
    from test_ui_lifecycle import settle

    monkeypatch.setattr(ctk.ScalingTracker, 'get_window_dpi_scaling',
                        classmethod(lambda cls, window: scale))
    root = ctk.CTk()
    root.minsize(1, 1)
    root.geometry('640x360')
    dialogs = []
    monkeypatch.setattr(overlap_tab.messagebox, 'showinfo', lambda *args, **kwargs: dialogs.append(args))
    results = sample_results()
    omitted = {'pipeline_1': 'A', 'pipeline_2': 'B', 'visualization_schema_version': 1,
               'visualization_status': 'omitted', 'visualization_polygons': [],
               'diagnostics': [{'code': 'corridor_buffer_limit'}]}
    for scope in [results, *results['geography']['states']]:
        scope['overlap_analysis'] = {'bundled_sections': [omitted], 'savings_miles': 0}
    try:
        show(root, version='test', current_file='sample.kmz', current_results=results,
             on_export=lambda: None, on_reanalyze=lambda: None, on_new_file=lambda: None,
             on_exit=lambda: None, on_open_corridor=lambda *args: None)
        pages = next(w for w in root.winfo_children() if isinstance(w, ResultPages))
        pages.set('Overlap Analysis')
        for name in ('Combined', 'Texas'):
            pages.scope_selector.set(name)
            pages.scope_selector.event_generate('<<ComboboxSelected>>')
            settle(root, .3)
            table = next(w for w in pages.pages['Overlap Analysis'].winfo_children()
                         if isinstance(w, CorridorTable))
            factor = ctk.ScalingTracker.get_widget_scaling(table)
            assert table.tree.winfo_height() >= 65 * factor, {
                'scope': name, 'table': table.winfo_height(), 'tree': table.tree.winfo_height(),
                'explanation': table.map_explanation.winfo_height(), 'scale': factor}
            assert table.tree.bbox(table.tree.get_children()[0])
            assert_inside(table.map_details, table.navigation)
            assert not table.map_explanation.winfo_ismapped()
            activate_for_keyboard(root)
            table.map_details.focus_set()
            settle(root)
            assert root.focus_get() is table.map_details
            count = len(dialogs)
            table.map_details.event_generate('<Return>')
            settle(root)
            assert len(dialogs) == count + 1  # Exactly one activation, including native bindings.
            assert 'drawing complexity' in dialogs[-1][1]
            table.tree.focus_set()
            settle(root)
            table.tree.event_generate('<Return>')
            settle(root)
            assert len(dialogs) == count + 2
            table.map_details.invoke()
            assert len(dialogs) == count + 3
        # Enlarging the window restores inline details and moves focus off the
        # now-hidden compact action, so subsequent Tab presses remain usable.
        table.map_details.focus_set()
        settle(root)
        root.geometry('1000x650')
        settle(root, .3)
        assert table.map_explanation.winfo_ismapped()
        assert not table.map_details.winfo_ismapped()
        assert root.focus_get() is table.tree
    finally:
        root.destroy()
