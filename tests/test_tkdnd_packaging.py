"""Package exactly the native directory selected by tkinterdnd2 at runtime."""
import os
from pathlib import Path, PurePosixPath
import platform
import runpy
from types import SimpleNamespace

import pytest
from PyInstaller.utils import hooks
from tkinterdnd2 import TkinterDnD


HOOK = Path(__file__).resolve().parents[1] / 'scripts/pyinstaller_hooks/hook-tkinterdnd2.py'


@pytest.fixture
def collected_data():
    # Collect the actual installed wheel before simulating a different host.
    return list(set(hooks.collect_data_files('tkinterdnd2') + hooks.collect_dynamic_libs('tkinterdnd2')))


def configure_platform(monkeypatch, system, machine, process_arch):
    monkeypatch.setattr(platform, 'system', lambda: system)
    monkeypatch.setattr(platform, 'machine', lambda: machine)
    if process_arch is None:
        monkeypatch.delenv('PROCESSOR_ARCHITECTURE', raising=False)
    else:
        monkeypatch.setenv('PROCESSOR_ARCHITECTURE', process_arch)


def platform_folder(destination):
    parts = PurePosixPath(str(destination).replace('\\', '/')).parts
    return parts[2] if parts[:2] == ('tkinterdnd2', 'tkdnd') and len(parts) > 2 else None


@pytest.mark.parametrize('system,machine,process_arch,expected,suffix', [
    ('Darwin', 'arm64', 'AMD64', 'osx-arm64', '.dylib'),
    ('Darwin', 'x86_64', None, 'osx-x64', '.dylib'),
    ('Windows', 'ARM64', 'AMD64', 'win-x64', '.dll'),
    ('Windows', 'AMD64', 'x86', 'win-x86', '.dll'),
    ('Windows', 'ARM64', 'ARM64', 'win-arm64', '.dll'),
    ('Windows', 'AMD64', None, 'win-x64', '.dll'),
    ('Linux', 'aarch64', 'AMD64', 'linux-arm64', '.so'),
    ('Linux', 'x86_64', None, 'linux-x64', '.so'),
])
def test_hook_matches_runtime_and_keeps_complete_selected_data(
        monkeypatch, tmp_path, collected_data, system, machine, process_arch, expected, suffix):
    sentinel = tmp_path / 'license.txt'
    sentinel.write_text('Unrelated package data must remain.')
    unrelated = (str(sentinel), 'tkinterdnd2')
    shared = (str(sentinel), os.path.join('tkinterdnd2', 'tkdnd'))
    inputs = [*collected_data, unrelated, shared]
    monkeypatch.setattr(hooks, 'collect_data_files', lambda name: list(inputs))
    monkeypatch.setattr(hooks, 'collect_dynamic_libs', lambda name: [])
    configure_platform(monkeypatch, system, machine, process_arch)
    result = runpy.run_path(str(HOOK))['datas']
    selected = [pair for pair in result if platform_folder(pair[1])]
    assert selected and {platform_folder(pair[1]) for pair in selected} == {expected}
    assert set(selected) == {pair for pair in inputs if platform_folder(pair[1]) == expected}
    names = {Path(source).name for source, _ in selected}
    assert {'pkgIndex.tcl', 'tkdnd.tcl'} <= names
    assert any(name.startswith('libtkdnd') and name.endswith(suffix) for name in names)
    assert unrelated in result and shared in result

    # Exercise the installed loader's selection without loading foreign code.
    # This catches divergence if tkinterdnd2 changes its architecture mapping.
    calls = []
    def call(*args):
        calls.append(args)
        return '8.6' if args == ('info', 'tclversion') else 'test-version'
    monkeypatch.setattr(TkinterDnD, 'TkdndVersion', None)
    TkinterDnD._require(SimpleNamespace(tk=SimpleNamespace(call=call)))
    runtime_folder = Path(next(args[2] for args in calls if args[:2] == ('lappend', 'auto_path'))).name
    assert runtime_folder == expected


@pytest.mark.parametrize('missing', ['directory', 'library', 'pkgIndex.tcl', 'tkdnd.tcl'])
def test_missing_selected_runtime_fails_build(monkeypatch, collected_data, missing):
    def keep(pair):
        source, destination = pair
        if platform_folder(destination) != 'osx-arm64':
            return True
        name = Path(source).name
        return missing != 'directory' and not (
            (missing == 'library' and name.endswith('.dylib')) or name == missing)
    monkeypatch.setattr(hooks, 'collect_data_files', lambda name: [pair for pair in collected_data if keep(pair)])
    monkeypatch.setattr(hooks, 'collect_dynamic_libs', lambda name: [])
    configure_platform(monkeypatch, 'Darwin', 'arm64', None)
    with pytest.raises(RuntimeError, match='missing the osx-arm64'):
        runpy.run_path(str(HOOK))


def test_unsupported_architecture_fails_build(monkeypatch):
    configure_platform(monkeypatch, 'Darwin', 'riscv64', None)
    with pytest.raises(RuntimeError, match='does not support Darwin/riscv64'):
        runpy.run_path(str(HOOK))


def test_actual_host_collection_keeps_one_native_directory():
    result = runpy.run_path(str(HOOK))['datas']
    assert len({platform_folder(destination) for _, destination in result if platform_folder(destination)}) == 1
    assert all(Path(source).is_file() for source, _ in result)
