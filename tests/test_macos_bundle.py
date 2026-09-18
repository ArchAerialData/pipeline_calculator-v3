"""Native header, architecture and containment gates for pre-sign app bundles."""
from pathlib import Path
import json
import os
import plistlib
import shutil
import stat
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.macos import validate_bundle as bundle


def thin(architecture='arm64', minimum=(14, 0, 0), *, legacy=False, platform=1):
    packed = minimum[0] << 16 | minimum[1] << 8 | minimum[2]
    command = (struct.pack('<IIII', 0x24, 16, packed, 15 << 16) if legacy else
               struct.pack('<IIIIII', 0x32, 24, platform, packed, 15 << 16, 0))
    return struct.pack('<IiiIIIII', 0xFEEDFACF, bundle.CPU_TYPES[architecture],
                       0, 2, 1, len(command), 0, 0) + command


def fat():
    slices = [thin('x86_64', (13, 0, 0)), thin('arm64', (14, 2, 1))]
    body = struct.pack('>II', 0xCAFEBABE, 2)
    for index, (architecture, data) in enumerate(zip(('x86_64', 'arm64'), slices), 1):
        body += struct.pack('>iiIII', bundle.CPU_TYPES[architecture], 0, index*4096, len(data), 12)
    for index, data in enumerate(slices, 1):
        body += b'\0' * (index*4096-len(body)) + data
    return body


def app_fixture(tmp_path, data=None):
    app = tmp_path / 'Example.app'
    executable = app / 'Contents/MacOS/Example'
    executable.parent.mkdir(parents=True)
    executable.write_bytes(data or thin())
    info = app / 'Contents/Info.plist'
    info.write_bytes(plistlib.dumps({'CFBundleExecutable': 'Example', 'CFBundleVersion': '4.26'}))
    info.chmod(0o644)
    return app, executable, info


def real_parser():
    # Present automatically on macOS through PyInstaller. Other hosts can run
    # these same fixture bytes by installing macholib (it is platform neutral).
    pytest.importorskip('macholib')


@pytest.mark.parametrize('architecture,expected', [('x86_64', (13, 0, 0)), ('arm64', (14, 2, 1))])
def test_real_fat_headers_use_requested_architecture_not_minimum_of_all_slices(tmp_path, architecture, expected):
    real_parser()
    path = tmp_path / 'universal'
    path.write_bytes(fat())
    assert bundle.read_minimum(path, architecture) == expected


@pytest.mark.parametrize('legacy', [True, False])
def test_real_thin_headers_support_both_macos_minimum_commands(tmp_path, legacy):
    real_parser()
    path = tmp_path / 'native'
    path.write_bytes(thin('x86_64', (13, 3, 1), legacy=legacy))
    assert bundle.read_minimum(path, 'x86_64') == (13, 3, 1)
    with pytest.raises(ValueError, match='arm64 slice'):
        bundle.read_minimum(path, 'arm64')


def test_real_bundle_stamps_largest_dependency_minimum_and_manifest(tmp_path):
    real_parser()
    app, _, info = app_fixture(tmp_path, thin(minimum=(11, 0, 0)))
    dependency = app / 'Contents/Frameworks/example.dylib'
    dependency.parent.mkdir()
    dependency.write_bytes(thin(minimum=(14, 2, 1)))
    output = tmp_path / 'checks/minimum.json'
    mode_before = stat.S_IMODE(info.stat().st_mode)
    result = bundle.validate_bundle(app, 'arm64', output=output)
    assert result['required_minimum_macos'] == '14.2.1'
    assert result['mach_o_file_count'] == 2
    assert json.loads(output.read_text()) == result
    assert plistlib.loads(info.read_bytes()) == {'CFBundleExecutable': 'Example',
        'CFBundleVersion': '4.26', 'LSMinimumSystemVersion': '14.2.1'}
    assert stat.S_IMODE(info.stat().st_mode) == mode_before
    assert dependency.read_bytes() == thin(minimum=(14, 2, 1))


@pytest.mark.parametrize('commands,message', [([], 'exactly one'),
    ([(0x32, dict(platform=2, minos=14 << 16))], 'another Apple platform'),
    ([(0x32, dict(platform=1, minos=0))], 'invalid'),
    ([(0x32, dict(platform=1, minos=14 << 16)), (0x24, dict(version=13 << 16))], 'exactly one')])
def test_invalid_or_conflicting_load_commands_fail_closed(commands, message):
    header = SimpleNamespace(commands=[(SimpleNamespace(cmd=cmd), SimpleNamespace(**data), None)
                                       for cmd, data in commands])
    with pytest.raises(ValueError, match=message):
        bundle.minimum_from_header(header)


def test_one_bad_dependency_prevents_plist_and_manifest_changes(tmp_path, monkeypatch):
    app, _, info = app_fixture(tmp_path)
    bad = app / 'Contents/MacOS/bad-library'
    bad.write_bytes(thin())
    original = info.read_bytes()
    def reader(path, architecture):
        if path.name == 'bad-library':
            raise ValueError('missing architecture')
        return (14, 0, 0)
    monkeypatch.setattr(bundle, 'read_minimum', reader)
    output = tmp_path / 'manifest.json'
    with pytest.raises(ValueError, match='bad-library.*missing architecture'):
        bundle.validate_bundle(app, 'arm64', output=output)
    assert info.read_bytes() == original and not output.exists()


def symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f'Test host does not permit native symlinks: {error}')


def test_internal_link_cycles_are_followed_without_duplicate_files(tmp_path):
    app, executable, info = app_fixture(tmp_path)
    symlink_or_skip(app / 'alias', executable)
    symlink_or_skip(app / 'loop', app, directory=True)
    assert set(bundle.bundle_files(app.resolve())) == {executable.resolve(), info.resolve()}


@pytest.mark.parametrize('outside', ['sibling', 'missing'])
def test_external_or_broken_links_fail_without_mutation(tmp_path, outside):
    app, _, info = app_fixture(tmp_path)
    original = info.read_bytes()
    target = tmp_path / (app.name + '-sibling')
    if outside == 'sibling':
        target.write_bytes(thin())
    symlink_or_skip(app / 'bad-link', target)
    with pytest.raises((ValueError, OSError)):
        bundle.validate_bundle(app, 'arm64', output=tmp_path/'manifest.json')
    assert info.read_bytes() == original


def test_plain_file_cannot_masquerade_as_bundle_executable(tmp_path):
    app, executable, info = app_fixture(tmp_path)
    executable.write_text('not a binary')
    with pytest.raises(ValueError, match='not a validated Mach-O'):
        bundle.validate_bundle(app, 'arm64', output=tmp_path/'manifest.json')


def test_architecture_defaults_to_actual_host(tmp_path, monkeypatch):
    app, _, _ = app_fixture(tmp_path)
    monkeypatch.setattr(bundle.platform, 'machine', lambda: 'x86_64')
    def reader(path, architecture):
        assert architecture == 'x86_64'
        return (13, 0, 0)
    monkeypatch.setattr(bundle, 'read_minimum', reader)
    result = bundle.validate_bundle(app, output=tmp_path/'manifest.json')
    assert result['architecture'] == 'x86_64'


def test_actual_cli_stamps_before_signing_and_reports_failure(tmp_path):
    real_parser()
    app, _, info = app_fixture(tmp_path)
    output = tmp_path / 'receipt.json'
    command = [sys.executable, str(Path(bundle.__file__)), str(app),
               '--architecture', 'arm64', '--output', str(output)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stdout + process.stderr
    assert 'minimum macOS 14.0.0' in process.stdout
    before = info.read_bytes()
    process = subprocess.run(command[:-1] + [str(app/'inside.json')],
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 1 and 'outside the app bundle' in process.stderr
    assert info.read_bytes() == before and not (app/'inside.json').exists()


@pytest.mark.parametrize('sign_exit,verify_exit', [(0, 0), (23, 0), (0, 24)])
def test_actual_build_finalization_reseals_after_stamping_and_propagates_failure(tmp_path, sign_exit, verify_exit):
    real_parser()
    import macholib
    bash = shutil.which('bash')
    if sys.platform == 'win32':
        candidate = Path('C:/Program Files/Git/bin/bash.exe')
        bash = str(candidate) if candidate.exists() else None
    if not bash:
        pytest.skip('Bash is required for build finalization boundary tests')
    def shell_path(path):
        value = path.resolve().as_posix()
        return '/'+value[0].lower()+value[2:] if sys.platform == 'win32' else value
    app, _, _ = app_fixture(tmp_path)
    (tmp_path/'build').mkdir()
    (tmp_path/'build/version.json').write_text(json.dumps({'numeric_version': '4.26', 'version': '4.26-dev.test'}))
    validator = tmp_path/'scripts/macos/validate_bundle.py'
    validator.parent.mkdir(parents=True)
    shutil.copyfile(bundle.__file__, validator)
    # Execute the actual finalization phase, after the expensive PyInstaller
    # boundary. Real fixture Mach-O headers and the real validator are retained.
    source = (Path(bundle.__file__).with_name('build_app.sh')).read_text()
    marker = '# Stamp numeric bundle fields and retain the full preview version separately.'
    assert source.count(marker) == 1
    script = tmp_path/'finalize.sh'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nAPP_PATH="$TEST_APP"\n'+
                      source[source.index(marker):], newline='\n')
    tools = tmp_path/'tools'
    tools.mkdir()
    python = tools/'python'
    python.write_text('#!/usr/bin/env bash\nexec "$TEST_PYTHON" "$@"\n', newline='\n')
    python.chmod(0o755)
    signer = tools/'codesign'
    signer.write_text('''#!/usr/bin/env bash
set -euo pipefail
"$TEST_PYTHON" - "$TEST_APP/Contents/Info.plist" <<'PY'
import plistlib, sys
with open(sys.argv[1], 'rb') as stream: info = plistlib.load(stream)
assert info['PipelineCalculatorVersion'] == '4.26-dev.test'
assert info['LSMinimumSystemVersion'] == '14.0.0'
PY
printf '%s\\n' "$1" >> "$TEST_EVENTS"
if [[ "$1" == '--force' ]]; then
  [[ "$2" == '--deep' && "$3" == '--sign' && "$4" == '-' && "$5" == "$TEST_APP" ]]
  exit "$SIGN_EXIT"
fi
[[ "$1" == '--verify' && "$2" == '--deep' && "$3" == '--strict' && "$4" == "$TEST_APP" ]]
exit "$VERIFY_EXIT"
''', newline='\n')
    signer.chmod(0o755)
    events = tmp_path/'events.txt'
    environment = dict(os.environ, TEST_TOOLS=shell_path(tools), TEST_SCRIPT=shell_path(script),
        TEST_PYTHON=shell_path(Path(sys.executable)), TEST_APP=shell_path(app),
        TEST_EVENTS=str(events), SIGN_EXIT=str(sign_exit), VERIFY_EXIT=str(verify_exit), ARTIFACT_ARCH='arm64',
        # Keep the real parser importable after switching to the isolated fixture
        # directory, including hosts which supply it through a relative PYTHONPATH.
        PYTHONPATH=str(Path(macholib.__file__).resolve().parents[1]))
    process = subprocess.run([bash, '-c', 'export PATH="$TEST_TOOLS:$PATH"; exec bash "$TEST_SCRIPT"'],
                             cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20)
    assert process.returncode == (sign_exit or verify_exit), process.stdout+process.stderr
    assert events.read_text().splitlines() == (['--force'] if sign_exit else ['--force', '--verify'])
    assert ('Build complete:' in process.stdout) == (process.returncode == 0)
