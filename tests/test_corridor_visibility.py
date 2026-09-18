"""Unavailable geography visuals cannot be launched by pointer or keyboard routes."""
from types import SimpleNamespace

import pytest

from pipeline_calculator.gui.actions.corridor_launch import launch_corridor
from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable


@pytest.mark.parametrize('decision', [
    {'visualization_status': 'omitted'}, {'clipped_polygons': []},
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
    sections.append({'pipeline_1': 'C', 'pipeline_2': 'D',
                     'visualization_schema_version': 1, 'visualization_status': 'ready'})
    launches = []
    table = CorridorTable(root, sections, lambda *args: launches.append(args))
    try:
        table.pack(fill='both', expand=True)
        root.update()
        first, second, missing = table.tree.get_children()
        for item in (first, missing):
            assert table.row_buttons[item].cget('text') == 'Map unavailable'
            assert table.row_buttons[item].instate(['disabled'])
            table.tree.selection_set(item)
            table._open_selected()
            table.row_buttons[item].invoke()
        assert launches == []
        assert table.row_buttons[second].cget('text') == 'View Corridor'
        assert table.row_buttons[second].instate(['!disabled'])
        table.tree.selection_set(second)
        table._open_selected()
        assert launches == [(sections[1], 2)]
    finally:
        root.destroy()
