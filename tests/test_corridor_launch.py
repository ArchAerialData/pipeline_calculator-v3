from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipeline_calculator.gui.actions import open_kml_action as action
from pipeline_calculator.gui.actions import corridor_launch as launcher


@pytest.mark.parametrize('geometry', [
    {}, {'visualization_polygons': None}, {'visualization_polygons': []},
    {'visualization_polygons': {}}, {'clipped_polygons': None},
    {'clipped_polygons': [], 'visualization_polygons': [{'outer': [[0, 0], [1, 0], [0, 1], [0, 0]], 'holes': []}]},
])
@pytest.mark.parametrize('use_dialog', [False, True])
def test_ready_without_canonical_polygons_disables_every_launch_route(geometry, use_dialog, monkeypatch):
    from pipeline_calculator.gui import config
    from pipeline_calculator.gui.tabs.overlap_tab import CorridorTable
    from pipeline_calculator.gui.dialogs import corridor_dialog

    section = dict(geometry, visualization_schema_version=1, visualization_status='ready',
                   bbox={'min_lon': -100, 'max_lon': -99, 'min_lat': 30, 'max_lat': 31})
    monkeypatch.setattr(config, 'SHOW_CORRIDOR_LAUNCH_DIALOG', use_dialog)
    with patch.object(launcher, 'DirectCorridorLaunch') as direct, \
         patch.object(corridor_dialog, 'CorridorDialog') as dialog, \
         patch.object(action, 'open_path') as opened:
        assert launcher.corridor_is_omitted(section)
        table = SimpleNamespace(item_map={'row': (section, 1)}, on_open_corridor=direct)
        CorridorTable._open(table, 'row')
        with pytest.raises(ValueError, match='map is unavailable'):
            launcher.launch_corridor(None, section, 1)
        direct.assert_not_called()
        dialog.assert_not_called()
        # The serializer also rejects these payloads before any file is opened.
        with pytest.raises(ValueError):
            action.create_and_launch_corridor(section, 1)
        opened.assert_not_called()


@pytest.mark.parametrize('clipped', [False, True])
def test_canonical_multipart_with_holes_remains_available(clipped):
    from xml.etree import ElementTree as ET

    polygons = [
        {'outer': [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
         'holes': [[[.5, .5], [.5, 1], [1, 1], [1, .5], [.5, .5]]]},
        {'outer': [[3, 0], [4, 0], [4, 1], [3, 1], [3, 0]], 'holes': []},
    ]
    section = {'visualization_schema_version': 1, 'visualization_status': 'ready',
               'visualization_polygons': polygons}
    if clipped:
        section.update(clipped_polygons=polygons, visualization_polygons=[])
    assert not launcher.corridor_is_omitted(section)
    root = ET.fromstring(action.build_overlap_corridor_kml(section, 1))
    ns = {'k': 'http://www.opengis.net/kml/2.2'}
    assert len(root.findall('.//k:Polygon', ns)) == 2
    assert len(root.findall('.//k:innerBoundaryIs', ns)) == 1


def test_legacy_geometry_without_canonical_decision_remains_available():
    section = {'bbox': {'min_lon': -100, 'max_lon': -99, 'min_lat': 30, 'max_lat': 31}}
    assert not launcher.corridor_is_omitted(section)


def test_nonzero_exit_and_timeout_preserve_path():
    for error in (FileNotFoundError('missing launcher'),
                  subprocess.CalledProcessError(7, ['open'], stderr=b'no association'),
                  subprocess.TimeoutExpired(['open'], 10)):
        with patch.object(action, 'open_path', side_effect=error):
            outcome = action.launch_saved_corridor('saved file.kml')
        assert outcome.path == 'saved file.kml' and outcome.status == 'failed' and outcome.error


@pytest.mark.parametrize('platform, command', [('darwin', 'open'), ('linux', 'xdg-open')])
def test_subprocess_checks_exit_status_and_timeout(platform, command):
    with patch.object(action.sys, 'platform', platform), patch.object(action.subprocess, 'run') as run:
        action.open_path('spaces and unicode Ã©.kml')
    assert run.call_args.args[0] == [command, 'spaces and unicode Ã©.kml']
    assert run.call_args.kwargs == {'check': True, 'capture_output': True, 'timeout': 10}


def test_windows_failure_and_successful_request():
    with patch.object(action.sys, 'platform', 'win32'), patch.object(action.os, 'startfile', create=True) as start:
        assert action.launch_saved_corridor('a.kml').status == 'requested'
        start.side_effect = OSError('no handler')
        assert action.launch_saved_corridor('a.kml').status == 'failed'


def test_compatibility_wrapper_retains_file_and_write_failure():
    with patch.object(action, 'write_corridor_kml_tempfile', return_value='saved.kml'), \
         patch.object(action, 'open_path', side_effect=OSError('no handler')):
        with pytest.raises(action.CorridorLaunchError) as error:
            action.open_overlap_corridor({}, 1)
    assert error.value.path == 'saved.kml'
    with patch.object(action, 'write_corridor_kml_tempfile', side_effect=OSError('disk full')), \
         patch.object(action, 'open_path') as opened:
        with pytest.raises(OSError, match='disk full'):
            action.create_and_launch_corridor({}, 1)
        opened.assert_not_called()


@pytest.mark.native_gui
def test_native_recovery_retry_save_copy_and_close(tmp_path, monkeypatch):
    import customtkinter as ctk
    from pipeline_calculator.gui.dialogs import corridor_dialog as ui
    import time
    path = tmp_path / 'Ã© corridor.kml'
    path.write_text('<kml/>', encoding='utf-8')
    created, launched = [], []
    def create(*args):
        created.append(args)
        return action.LaunchOutcome(str(path), 'failed', 'no handler')
    monkeypatch.setattr(ui, 'create_and_launch_corridor', create)
    monkeypatch.setattr(ui, 'launch_saved_corridor', lambda p: launched.append(p) or action.LaunchOutcome(p, 'requested'))
    root = ctk.CTk(); root.withdraw()
    clipboard_before = None
    try:
        clipboard_before = root.clipboard_get()
    except Exception:
        pass
    dialog = ui.CorridorDialog(root, {}, 1)
    dialog.window.withdraw()
    try:
        assert dialog.done.wait(5)
        dialog._poll()
        assert 'could not be opened' in dialog.label.cget('text')
        dialog.copy_path()
        assert root.clipboard_get() == str(path)
        destination = tmp_path / 'kept.kml'
        monkeypatch.setattr(ui.filedialog, 'asksaveasfilename', lambda **k: str(destination))
        dialog.save_as()
        assert destination.read_bytes() == path.read_bytes()
        errors = []
        monkeypatch.setattr(ui.shutil, 'copyfile', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
        monkeypatch.setattr(ui.messagebox, 'showerror', lambda *a, **k: errors.append(a))
        dialog.save_as()
        assert errors and path.exists()
        dialog.retry()
        assert dialog.done.wait(5)
        dialog._poll()
        assert 'Opening requested' in dialog.label.cget('text')
        assert len(created) == 1 and launched == [str(path)]
    finally:
        dialog.close()
        root.clipboard_clear()
        if clipboard_before is not None:
            root.clipboard_append(clipboard_before)
        root.update_idletasks()
        root.destroy()
    assert path.exists()


@pytest.mark.native_gui
def test_backend_toggle_direct_open_failure_and_shutdown(monkeypatch):
    import customtkinter as ctk
    import threading
    import time
    from pipeline_calculator.gui import config
    from pipeline_calculator.gui.actions import corridor_launch as launcher
    from pipeline_calculator.gui.dialogs import corridor_dialog
    assert config.SHOW_CORRIDOR_LAUNCH_DIALOG is False
    root = ctk.CTk()
    root.withdraw()
    calls, errors = [], []
    monkeypatch.setattr(launcher.messagebox, 'showerror', lambda *a, **k: errors.append(a))
    monkeypatch.setattr(action, 'create_and_launch_corridor', lambda s, i: calls.append((s, i)) or action.LaunchOutcome('corridor.kml', 'requested'))
    before = root.winfo_children()
    root_destroyed = False
    try:
        job = launcher.launch_corridor(root, {'name': 'chosen'}, 42)
        assert job.done.wait(3)
        until = time.monotonic()+.2
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)
        assert calls == [({'name': 'chosen'}, 42)] and not errors
        assert root.winfo_children() == before and job.closed
        monkeypatch.setattr(config, 'SHOW_CORRIDOR_LAUNCH_DIALOG', True)
        monkeypatch.setattr(corridor_dialog, 'CorridorDialog', lambda *a: calls.append(a) or 'dialog')
        assert launcher.launch_corridor(root, {}, 3) == 'dialog'
        monkeypatch.setattr(config, 'SHOW_CORRIDOR_LAUNCH_DIALOG', False)
        monkeypatch.setattr(action, 'create_and_launch_corridor', lambda *a: action.LaunchOutcome('saved.kml', 'failed', 'No KML handler'))
        job = launcher.launch_corridor(root, {}, 4)
        assert job.done.wait(3)
        until = time.monotonic()+.2
        while time.monotonic() < until:
            root.update()
            time.sleep(.005)
        assert len(errors) == 1 and 'saved.kml' in errors[0][1]
        release = threading.Event()
        def delayed_launch(*args):
            release.wait(3)
            return action.LaunchOutcome('late.kml', 'requested')
        monkeypatch.setattr(action, 'create_and_launch_corridor', delayed_launch)
        job = launcher.launch_corridor(root, {}, 5)
        root.destroy()
        root_destroyed = True
        assert job.closed and job.poll_id is None
        release.set()
        assert job.done.wait(3)
    finally:
        if not root_destroyed:
            root.destroy()
