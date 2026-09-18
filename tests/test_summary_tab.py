"""Summary must keep incomplete estimates and real zero results distinct."""
import pytest
import customtkinter as ctk
import time

from pipeline_calculator.gui.layout import WrappedLabel
from pipeline_calculator.gui.tabs.summary_tab import SummaryView, InlinePair, number


def labels(widget):
    result = []
    for child in widget.winfo_children():
        if isinstance(child, ctk.CTkLabel):
            result.append(child.cget('text'))
        result.extend(labels(child))
    return result


@pytest.mark.native_gui
def test_summary_preserves_estimate_states():
    root = ctk.CTk()
    try:
        cases = [
            ({'analysis_complete': True, 'total_miles': 0, 'overlap_analysis': {
                'effective_total_miles': 0, 'savings_miles': 0, 'savings_percentage': 0}},
             '0.000', 'Mileage Removed:\n0.000 mi (0.0%)', False),
            ({'analysis_complete': False, 'overlap_analysis': None,
              'diagnostics': [{'level': 'error', 'message': 'Overlap calculation failed.'}]},
             'Unavailable', 'Overlap calculation failed.', True),
            ({'analysis_complete': True, 'pipelines': [{}], 'overlap_analysis': None},
             'Not applicable', 'Overlap analysis requires at least two pipelines.', False),
            ({'analysis_complete': True, 'overlap_analysis': {'effective_total_miles': float('nan')}},
             'Unavailable', 'Adjusted mileage was not produced for this run.', False),
        ]
        for changes, value, expected, incomplete in cases:
            data = {'pipelines': [{}, {}], 'total_miles': 10, 'diagnostics': [], **changes}
            view = SummaryView(root, data)
            view.pack(fill='both', expand=True)
            root.update()
            assert view.adjusted.value.cget('text') == value
            shown = '\n'.join(labels(view.inner))
            assert expected in shown
            assert ('Analysis incomplete.' in shown) == incomplete
            assert not view.details.winfo_manager()
            assert not view.details.winfo_children()
            view.toggle.invoke()
            assert 'Detection range: Not recorded' in '\n'.join(labels(view.details))
            view.destroy()
    finally:
        root.destroy()


def test_estimate_number_does_not_turn_unavailable_into_zero():
    for missing in (None, float('nan'), float('inf'), 'unknown'):
        assert number(missing) == 'Unavailable'
    assert number(0) == '0.000'
    assert number(2344.562591432312) == '2,344.563'


@pytest.mark.native_gui
def test_summary_typography_across_viewports():
    """Exercise real font metrics and DPI changes, including future mobile widths."""
    ctk.ScalingTracker.get_window_dpi_scaling = classmethod(lambda cls, window: 1)
    root = ctk.CTk()
    root.minsize(1, 1)
    root.maxsize(5000, 3000)
    errors = []
    root.report_callback_exception = lambda *args: errors.append(str(args))
    data = {'total_miles': 2344.563, 'analysis_complete': True, 'pipelines': [{}, {}],
            'overlap_analysis': {'effective_total_miles': 2245.209,
                                 'savings_miles': 99.354, 'savings_percentage': 4.2,
                                 'bundled_sections': [{}] * 95}}
    view = SummaryView(root, data)
    view.pack(fill='both', expand=True, padx=16, pady=16)

    def settle(duration=.35):
        until = time.monotonic() + duration
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)
        assert not errors, errors

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    try:
        # Logical client sizes, independent of physical monitor diagonal.
        # Windows desktop/widescreen, laptop at 2x, tablet, phone, then desktop again.
        previous_scale = None
        for width, height, scale in [(1920, 1040, 1), (2560, 1360, 1), (2752, 1080, 1.25),
                                      (1280, 800, 2), (1440, 900, 2), (1536, 960, 2),
                                      (768, 1024, 1), (390, 844, 1), (320, 640, 1),
                                      (1440, 900, 1.5)]:
            if scale != previous_scale:
                ctk.set_widget_scaling(scale)
                ctk.set_window_scaling(scale)
                settle(1.1)  # CTk releases its native size lock 1000 ms after changing DPI.
                previous_scale = scale
            root.geometry(f'{width}x{height}+-8000+0')
            settle()
            assert abs(root.winfo_width()/scale - width) <= 2
            assert view.original.cget('fg_color') == view.adjusted.cget('fg_color')
            for card in (view.original, view.adjusted):
                assert card.unit.grid_info()['row'] == card.value.grid_info()['row']
                assert card.unit.winfo_rootx() >= card.value.winfo_rootx() + card.value.winfo_width()
                baselines = [label._label.winfo_rooty() + label._label.winfo_height() -
                             int(root.tk.call('font', 'metrics', label._label.cget('font'), '-descent'))
                             for label in (card.value, card.unit)]
                assert abs(baselines[0] - baselines[1]) <= 2, (width, scale, baselines)
                for fact in card.facts.winfo_children():
                    if isinstance(fact, InlinePair):
                        assert fact.label.cget('font').cget('weight') == 'bold'
                        assert fact.value.cget('font').cget('weight') == 'normal'
                for label in descendants(card):
                    if isinstance(label, ctk.CTkLabel):
                        assert label._label.winfo_reqwidth() <= label.winfo_width()+2, (width, scale, label.cget('text'))
                        assert label._label.winfo_reqheight() <= label.winfo_height()+2, (width, scale, label.cget('text'), 'height')
            if width >= 1280:
                assert view.original.grid_info()['row'] == view.adjusted.grid_info()['row']
                assert abs(view.original.title.winfo_rooty() - view.adjusted.title.winfo_rooty()) <= 1
                assert abs(view.original.value.winfo_rooty() - view.adjusted.value.winfo_rooty()) <= 1
            else:
                assert view.original.grid_info()['row'] != view.adjusted.grid_info()['row']
            view.toggle.invoke()
            settle()
            view.toggle.invoke()
    finally:
        root.destroy()
        ctk.set_widget_scaling(1)
        ctk.set_window_scaling(1)
