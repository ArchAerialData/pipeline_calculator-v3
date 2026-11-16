from pathlib import Path
import hashlib
import runpy
import sys

import pytest

from pipeline_calculator.smoke import _check_gis_packaging


def test_offline_gis_smoke_exercises_real_native_readers(tmp_path, monkeypatch):
    monkeypatch.setenv('PROJ_NETWORK', 'OFF')
    report = _check_gis_packaging(tmp_path)
    for key in ('shapefile', 'direct_zip', 'gpkg_layer_selection', 'geojson', 'projected_crs', 'captured_source'):
        assert report[key] is True
    assert report['drivers'] == ['ESRI Shapefile', 'GPKG', 'GeoJSON']
    assert report['gdal'] and report['pyogrio']


def test_pyinstaller_hook_preserves_driver_data_and_wheel_libraries():
    root = Path(__file__).resolve().parents[1]
    hook = runpy.run_path(str(root / 'scripts/pyinstaller_hooks/hook-pyogrio.py'))
    assert any('gdal_data' in source for source, _ in hook['datas'])
    assert any('proj_data' in source and source.endswith('proj.db') for source, _ in hook['datas'])
    if sys.platform == 'win32':
        assert any('gdal-' in Path(source).name and destination == 'pyogrio.libs'
                   for source, destination in hook['binaries'])


@pytest.mark.skipif(sys.platform != 'win32', reason='Delvewheel DLL directory lookup is Windows-specific')
def test_shared_msvc_runtime_is_collected_in_each_wheels_own_directory():
    from PyInstaller.building.datastruct import normalize_toc
    root = Path(__file__).resolve().parents[1]
    hooks = {
        'pandas.libs': runpy.run_path(str(root / 'scripts/pyinstaller_hooks/hook-pandas._libs.py')),
        'pyogrio.libs': runpy.run_path(str(root / 'scripts/pyinstaller_hooks/hook-pyogrio.py')),
    }
    runtimes = {}
    for destination, hook in hooks.items():
        owned = {Path(source).name.lower(): Path(source) for source, target in hook['binaries']
                 if target == destination and Path(source).name.lower().startswith('msvcp140-')}
        assert owned, f'{destination} must be available before that package is imported'
        runtimes[destination] = owned
    common = runtimes['pandas.libs'].keys() & runtimes['pyogrio.libs'].keys()
    entries = [(str(Path(target) / Path(source).name), source, 'BINARY')
               for hook in hooks.values() for source, target in hook['binaries']]
    bundled = {name.lower().replace('\\', '/') for name, _, _ in normalize_toc(entries)}
    # The installed Windows wheels reproduce the original basename collision.
    # Each import must find its own copy even though the bytes are identical.
    for name in common:
        pandas_runtime = runtimes['pandas.libs'][name]
        gdal_runtime = runtimes['pyogrio.libs'][name]
        assert pandas_runtime.parent != gdal_runtime.parent
        assert hashlib.sha256(pandas_runtime.read_bytes()).digest() == hashlib.sha256(gdal_runtime.read_bytes()).digest()
        assert {f'pandas.libs/{name}', f'pyogrio.libs/{name}'} <= bundled
