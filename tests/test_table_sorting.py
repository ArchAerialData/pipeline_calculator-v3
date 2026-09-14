import time
from tkinter import ttk
import customtkinter as ctk
import pytest

from pipeline_calculator.gui.sorting import sort_records


def test_numeric_sort_precision_missing_and_stability():
    rows = [('a', 10), ('b', 2), ('c', None), ('d', float('nan')), ('e', 2.000001), ('f', 2)]
    assert [r[0] for r in sort_records(rows, lambda r: r[1], numeric=True)] == ['b', 'f', 'e', 'a', 'c', 'd']
    assert [r[0] for r in sort_records(rows, lambda r: r[1], numeric=True, descending=True)] == ['a', 'e', 'b', 'f', 'c', 'd']
    assert sort_records(['10', '2', 'A', ''], lambda x: x, numeric=True) == ['2', '10', 'A', '']
    assert sort_records(['zebra', 'Alpha', 'beta'], lambda x: x) == ['Alpha', 'beta', 'zebra']


@pytest.mark.native_gui
def test_sort_headers_totals_pagination_and_corridor_identity():
    from pipeline_calculator.gui.tabs import pipelines_tab
    from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable
    ctk.ScalingTracker.get_window_dpi_scaling = classmethod(lambda cls, window: 1)
    root = ctk.CTk()
    root.geometry('1200x650')
    frame = ctk.CTkFrame(root)
    frame.pack(fill='both', expand=True)
    def widgets(node):
        for child in node.winfo_children():
            yield child
            yield from widgets(child)
    def settle():
        until = time.monotonic()+.2
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)
    def click_heading(tree, column):
        tree.xview_moveto(0)
        settle()
        x = sum(tree.column(c, 'width') for c in tree['columns'][:tree['columns'].index(column)]) + 12
        tree.event_generate('<ButtonPress-1>', x=x, y=10)
        tree.event_generate('<ButtonRelease-1>', x=x, y=10)
        settle()
    try:
        pipelines_tab.create(frame, {'pipelines': [
            {'Placemark_ID': 'Z10', 'Name': 'Beta', 'Shape_Length': 2.000002, 'pipelinelength': 10},
            {'Placemark_ID': 'A2', 'Name': 'alpha', 'Shape_Length': 2.000001, 'pipelinelength': 2}],
            'total_meters': 4.000003, 'total_miles': 12})
        tree = next(w for w in widgets(frame) if isinstance(w, ttk.Treeview))
        settle()
        assert all(str(tree.sorter.images['neutral']) in str(tree.heading(c, 'image')) for c in tree['columns'])
        assert tree.sorter.column is None
        total = tree.get_children()[-1]
        for column, expected_id in [('Placemark ID', 'A2'), ('Name', 'A2'), ('Length (m)', 'A2'), ('Length (miles)', 'A2')]:
            click_heading(tree, column)
            assert str(tree.set(tree.get_children()[0], 'Placemark ID')) == expected_id
            assert str(tree.sorter.images['ascending']) in str(tree.heading(column, 'image'))
            assert tree.get_children()[-1] == total
            click_heading(tree, column)
            assert str(tree.set(tree.get_children()[0], 'Placemark ID')) == 'Z10'
            assert str(tree.sorter.images['descending']) in str(tree.heading(column, 'image'))
            assert tree.get_children()[-1] == total
        assert sum(str(tree.sorter.images['descending']) in str(tree.heading(c, 'image')) for c in tree['columns']) == 1
        frame.destroy()
        opened = []
        sections = [{'pipeline_1': f'P{i:02}', 'pipeline_2': 'B', 'bundled_length_miles': 45-i,
                     'average_separation': i % 3} for i in range(45)]
        table = CorridorTable(root, sections, lambda section, index: opened.append((section, index)))
        table.pack(fill='both', expand=True)
        settle()
        assert all(table.tree.heading(c, 'image') for c in table.sorter.titles)
        assert not table.tree.heading('Action', 'image')
        click_heading(table.tree, 'Length (miles)')
        assert table.ordered_sections[0] == (45, sections[44])
        first = table.tree.get_children()[0]
        table.row_buttons[first].invoke()
        assert opened[-1][0] is sections[44] and opened[-1][1] == 45
        table.next_button.invoke()
        settle()
        first = table.tree.get_children()[0]
        assert table.item_map[first][1] == 25
        table.row_buttons[first].invoke()
        assert opened[-1][0] is sections[24] and opened[-1][1] == 25
        assert str(table.sorter.images['ascending']) in str(table.tree.heading('Length (miles)', 'image'))
        click_heading(table.tree, 'Length (miles)')
        assert table.page == 0 and table.ordered_sections[0][0] == 1
        click_heading(table.tree, 'Pipeline Pair')
        assert table.ordered_sections[0][0] == 1
        click_heading(table.tree, 'Avg Sep (m)')
        assert [r[1]['average_separation'] for r in table.ordered_sections] == sorted(s['average_separation'] for s in sections)
        assert [s['bundled_length_miles'] for s in sections] == list(range(45, 0, -1))
    finally:
        root.destroy()
