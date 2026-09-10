"""Check normal startup without update() or forced visibility masking failures."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
if len(sys.argv) > 1 and sys.argv[1] == 'legacy':
    from pipeline_calculator_v3 import PipelineCalculatorGUI
else:
    from pipeline_calculator.gui.main_window import PipelineCalculatorGUI

app = PipelineCalculatorGUI()
states = []


def sample():
    states.append((app.root.state(), bool(app.root.winfo_viewable())))


try:
    app.root.after(350, sample)
    app.root.after(1600, sample)
    app.root.after(1800, app.root.quit)
    # Match normal application startup: no update() call before mainloop().
    app.run()
    assert states == [('normal', True), ('normal', True)], states
    print('Normal startup remains visible')
finally:
    app.close()
