"""Check the macOS wheel layout without executing foreign native code."""
from pathlib import Path
import plistlib
import runpy
import shutil
import struct

import pytest

from scripts.macos import validate_bundle as bundle


def arm64_binary(minimum=12):
    # A minimal real Mach-O header lets the actual minimum-OS validator run.
    command = struct.pack('<IIIIII', 0x32, 24, 1, minimum << 16, 15 << 16, 0)
    return struct.pack('<IiiIIIII', 0xFEEDFACF, bundle.CPU_TYPES['arm64'],
                       0, 2, 1, len(command), 0, 0) + command


def test_macos_gdal_hidden_directory_survives_hook_bundle_links_and_validation(tmp_path, monkeypatch):
    pytest.importorskip('macholib')  # Installed by PyInstaller on native macOS.
    import PyInstaller.utils.hooks as hooks
    from PyInstaller.building.osx import BUNDLE

    wheel = tmp_path / 'wheel'
    package = wheel / 'pyogrio'
    files = {
        '__init__.py': b'',
        '_io.abi3.so': arm64_binary(),
        '.dylibs/libgdal.38.3.12.4.dylib': arm64_binary(),
        'gdal_data/header.dxf': b'GDAL resource',
        'proj_data/proj.db': b'PROJ resource',
    }
    for name, content in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    # The hook must inspect files without importing a platform-specific binary.
    # These are the actual package-discovery boundaries used by PyInstaller.
    monkeypatch.setattr(hooks, 'get_package_paths', lambda name: (str(wheel), str(package)))
    monkeypatch.setattr(hooks, 'get_all_package_paths', lambda name: [str(package)])
    root = Path(__file__).resolve().parents[1]
    hook = runpy.run_path(str(root / 'scripts/pyinstaller_hooks/hook-pyogrio.py'))
    gdal_source = package / '.dylibs/libgdal.38.3.12.4.dylib'
    assert [(Path(source), Path(destination)) for source, destination in hook['binaries']] == [
        (gdal_source, Path('pyogrio/.dylibs'))]
    assert 'pyogrio._io' in hook['hiddenimports']
    assert {Path(source).relative_to(package).as_posix() for source, _ in hook['datas']} == {
        'gdal_data/header.dxf', 'proj_data/proj.db'}

    executable = tmp_path / 'Example'
    executable.write_bytes(arm64_binary(minimum=11))
    toc = [('Example', str(executable), 'EXECUTABLE'),
           ('pyogrio/_io.abi3.so', str(package / '_io.abi3.so'), 'EXTENSION')]
    for key, kind in [('datas', 'DATA'), ('binaries', 'BINARY')]:
        toc.extend((str(Path(destination) / Path(source).name), source, kind)
                   for source, destination in hook[key])

    # Use the real PyInstaller relocation, including its codesign-compatible
    # .dylibs rename and links between Resources and Frameworks.
    relocated = object.__new__(BUNDLE)._process_bundle_toc(toc)
    app = tmp_path / 'Example.app'
    for destination, source, kind in relocated:
        if kind == 'SYMLINK':
            continue
        target = app / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for destination, source, kind in relocated:
        if kind != 'SYMLINK':
            continue
        target = app / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        # Windows requires the directory flag when creating a directory link.
        # Resolve only its type; retain PyInstaller's original relative target.
        is_directory = (target.parent / source).suffix not in ('.so', '.dylib')
        try:
            target.symlink_to(source, target_is_directory=is_directory)
        except OSError as error:
            pytest.skip(f'Test host does not permit native symlinks: {error}')
    (app / 'Contents/Info.plist').write_bytes(plistlib.dumps({'CFBundleExecutable': 'Example'}))
    report = bundle.validate_bundle(app, 'arm64', output=tmp_path / 'bundle.json')
    assert report['mach_o_file_count'] == 3
    assert report['required_minimum_macos'] == '12.0.0'
    for base in ('Frameworks', 'Resources'):
        linked_package = app / 'Contents' / base / 'pyogrio'
        assert (linked_package / '.dylibs' / gdal_source.name).read_bytes() == gdal_source.read_bytes()
        assert (linked_package / 'proj_data/proj.db').read_bytes() == b'PROJ resource'
        assert (linked_package / 'gdal_data/header.dxf').read_bytes() == b'GDAL resource'
    assert all(path.is_relative_to(app.resolve()) for path in bundle.bundle_files(app.resolve()))
