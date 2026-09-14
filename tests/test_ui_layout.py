from pathlib import Path
import sys

import pytest
from pipeline_calculator.gui.window import fitted_geometry
from scripts.validation.gui_process import run_gui


@pytest.mark.parametrize('area', [(0, 0, 1024, 728), (0, 0, 1920, 1040),
                                  (-2560, -100, 0, 1340), (0, -2160, 3840, 0)])
@pytest.mark.parametrize('scale', [1, 1.25, 1.5, 2, 2.5])
def test_startup_fits_work_area(area, scale):
    width, height, x, y = fitted_geometry(area, scale)
    assert width > 0 and height > 0
    assert x >= area[0] and y >= area[1]
    assert x + (width + 16)*scale <= area[2]
    assert y + (height + 48)*scale <= area[3]


@pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows layout verification')
@pytest.mark.parametrize('scale,width,height,impl', [
    (1, 440, 340, 'new'), (1.25, 640, 480, 'new'), (1.5, 800, 600, 'new'),
    (2, 480, 320, 'new'), (2.5, 1000, 720, 'new'), (2.5, 512, 288, 'new'),
    (2.5, 408, 288, 'new'),
    (1, 1800, 900, 'new'),
    (1.25, 640, 480, 'legacy')])
def test_native_layout(scale, width, height, impl):
    result = run_gui([sys.executable, str(Path(__file__).with_name('ui_layout_probe.py')),
                             str(scale), str(width), str(height), '-', impl],
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "passed"' in result.stdout


@pytest.mark.skipif(sys.platform != 'win32', reason='Native Windows startup verification')
@pytest.mark.parametrize('impl', ['new', 'legacy'])
def test_normal_mainloop_startup_stays_visible(impl):
    result = run_gui([sys.executable, str(Path(__file__).with_name('ui_startup_probe.py')), impl],
                            timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Normal startup remains visible' in result.stdout
