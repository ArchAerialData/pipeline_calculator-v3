"""Corridor layout recovery survives a tree squeezed out of its viewport."""
import customtkinter as ctk
import pytest

from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable
from test_ui_lifecycle import settle


@pytest.mark.native_gui
def test_tree_unmap_during_shrink_recovers_and_hidden_table_cancels_callbacks(monkeypatch):
    root = ctk.CTk()
    root.minsize(1, 1)
    root.geometry('900x400')
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    row = {'pipeline_1': 'A', 'pipeline_2': 'B', 'visualization_status': 'omitted',
           'diagnostics': [{'code': 'corridor_buffer_limit'}]}
    table = CorridorTable(root, [row], lambda *args: pytest.fail('An omitted map was launched'))
    table.pack(fill='both', expand=True)
    tree_unmaps = []
    table.tree.bind('<Unmap>', lambda event: tree_unmaps.append(event.widget), add='+')
    try:
        settle(root, .3)
        assert table.map_explanation.winfo_ismapped()
        assert not table._details_compact
        root.geometry('450x90')
        settle(root, .4)
        # The large inline explanation temporarily leaves no room for the tree.
        # Its Unmap must not cancel the queued layout which makes room again.
        assert tree_unmaps, 'The regression must exercise a real native tree Unmap'
        assert table._details_compact
        assert not table.map_explanation.winfo_ismapped()
        assert table.map_details.winfo_ismapped()
        assert table.tree.winfo_ismapped()

        calls = []
        position = table._position_buttons

        def record_position():
            calls.append(table.winfo_viewable())
            position()

        monkeypatch.setattr(table, '_position_buttons', record_position)
        table._queue_position()
        pending = table._position_id
        assert pending is not None
        table.pack_forget()
        settle(root)
        assert table._position_id is None
        assert pending not in root.tk.splitlist(root.tk.call('after', 'info'))
        assert calls == []

        table.pack(fill='both', expand=True)
        settle(root, .2)
        assert table.tree.winfo_ismapped()
        table._queue_position()
        pending = table._position_id
        assert pending is not None
        before_destroy = len(calls)
        table.destroy()
        settle(root)
        assert pending not in root.tk.splitlist(root.tk.call('after', 'info'))
        assert len(calls) == before_destroy
        assert errors == []
    finally:
        root.destroy()
