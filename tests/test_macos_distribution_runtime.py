"""Exercise runtime-only distribution gates without building another app."""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def shell_path(path):
    text = path.resolve().as_posix()
    return '/'+text[0].lower()+text[2:] if sys.platform == 'win32' else text


def executable(path, text):
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+text, newline='\n')
    path.chmod(0o755)


@pytest.fixture
def runtime_gate(tmp_path):
    bash = shutil.which('bash')
    if sys.platform == 'win32':
        candidate = Path('C:/Program Files/Git/bin/bash.exe')
        bash = str(candidate) if candidate.exists() else None
    if not bash:
        pytest.skip('Bash required for runtime distribution boundary tests')
    repo = tmp_path/'runtime workspace'
    script = repo/'scripts/ci/macos_verify_distribution.sh'
    script.parent.mkdir(parents=True)
    text = (ROOT/'scripts/ci/macos_verify_distribution.sh').read_text()
    # Substitute only the native machine-query boundary. All archive, version,
    # receipt and failure behavior runs through the actual helper unchanged.
    assert text.count('platform.machine()') == 1
    script.write_text(text.replace('platform.machine()', "os.environ['TEST_MACHINE']"), newline='\n')
    tools = repo/'tools'
    tools.mkdir()
    executable(tools/'sw_vers', '''
if [[ "$1" == '-productVersion' ]]; then printf '%s\n' "$TEST_OS";
elif [[ "$1" == '-buildVersion' ]]; then printf '%s\n' '26A5406e'; else exit 98; fi
''')
    executable(tools/'ditto', '''
printf 'extract\n' >> "$TEST_EVENTS"
[[ "$1" == '-x' && "$2" == '-k' ]] || exit 98
"$PYTHON_BIN" - "$3" "$4" <<'PY'
from pathlib import Path
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
''')
    executable(tools/'codesign', '''
printf 'signature\n' >> "$TEST_EVENTS"
[[ "$1" == '--verify' && "$2" == '--deep' && "$3" == '--strict' && -d "$4" ]] || exit 98
exit "$SIGNATURE_EXIT"
''')
    checker = repo/'scripts/validation/check_packaged_smoke.py'
    checker.parent.mkdir()
    checker.write_text('''import argparse, json, os
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('app', type=Path)
parser.add_argument('--expected-version', required=True)
parser.add_argument('--output-directory', type=Path, required=True)
args = parser.parse_args()
assert (args.app/'Contents/MacOS/Example').read_bytes() == b'EXACT ARCHIVED APP'
with Path(os.environ['TEST_EVENTS']).open('a') as stream: stream.write('smoke\\n')
args.output_directory.mkdir(parents=True)
(args.output_directory/'arguments.json').write_text(json.dumps({'version': args.expected_version, 'app': str(args.app)}))
raise SystemExit(int(os.environ['SMOKE_EXIT']))
''')
    archive = repo/'built app.zip'
    output = repo/'runtime receipt'
    events = repo/'events.txt'

    def run(*, actual_os='26.1', expected_os='26', machine='arm64',
            signature_exit=0, smoke_exit=0, embedded='4.26-dev.test', bad_entry=None):
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr('Pipeline_Calculator_v5.app/Contents/Info.plist',
                plistlib.dumps({'PipelineCalculatorVersion': embedded,
                                'CFBundleExecutable': 'Example', 'LSMinimumSystemVersion': '14.0.0'}))
            zipped.writestr('Pipeline_Calculator_v5.app/Contents/MacOS/Example', b'EXACT ARCHIVED APP')
            if bad_entry:
                zipped.writestr(bad_entry, 'should not extract')
        original = archive.read_bytes()
        env = dict(os.environ, PYTHON_BIN=shell_path(Path(sys.executable)),
                   TEST_TOOLS=shell_path(tools), TEST_EVENTS=str(events), TEST_SCRIPT=shell_path(script),
                   TEST_ARCHIVE=shell_path(archive), TEST_OUTPUT=shell_path(output),
                   TEST_MACHINE=machine, TEST_OS=actual_os, TEST_EXPECTED_OS=expected_os,
                   SIGNATURE_EXIT=str(signature_exit), SMOKE_EXIT=str(smoke_exit))
        process = subprocess.run([bash, '-c',
            'export PATH="$TEST_TOOLS:$PATH"; exec bash "$TEST_SCRIPT" "$TEST_ARCHIVE" '
            '"$TEST_EXPECTED_OS" "4.26-dev.test" "$TEST_OUTPUT"'],
            cwd=repo, env=env, capture_output=True, text=True, timeout=25)
        assert archive.read_bytes() == original
        receipt = json.loads((output/'runtime.json').read_text())
        assert receipt['archive_sha256'] == hashlib.sha256(original).hexdigest()
        return process, receipt, events.read_text().splitlines() if events.exists() else [], output
    return run


@pytest.mark.parametrize('major', ['26', '27'])
def test_same_archived_app_runs_on_requested_native_runtime(runtime_gate, major):
    process, receipt, events, output = runtime_gate(actual_os=major+'.0', expected_os=major)
    assert process.returncode == 0, process.stdout+process.stderr
    assert events == ['extract', 'signature', 'smoke']
    assert receipt['status'] == 'passed' and receipt['actual_os_version'] == major+'.0'
    assert receipt['architecture'] == 'arm64' and receipt['actual_os_build'] == '26A5406e'
    assert receipt['embedded_version'] == '4.26-dev.test'
    assert json.loads((output/'smoke/arguments.json').read_text())['version'] == '4.26-dev.test'


@pytest.mark.parametrize('arguments,message', [
    (dict(actual_os='26.0', expected_os='27'), 'actual runtime is 26.0'),
    (dict(machine='x86_64'), 'native arm64'),
    (dict(embedded='4.25'), 'Embedded version mismatch'),
    (dict(bad_entry='../escaped.txt'), 'Unexpected archive path')])
def test_wrong_runtime_architecture_version_or_archive_never_smokes(runtime_gate, arguments, message):
    process, receipt, events, output = runtime_gate(**arguments)
    assert process.returncode != 0 and message in process.stderr
    assert receipt['status'] == 'failed' and 'smoke' not in events
    assert 'signature' not in events and not (output/'smoke').exists()


@pytest.mark.parametrize('signature_exit,smoke_exit,expected_events', [
    (23, 0, ['extract', 'signature']), (0, 24, ['extract', 'signature', 'smoke'])])
def test_signature_or_smoke_failure_propagates(runtime_gate, signature_exit, smoke_exit, expected_events):
    process, receipt, events, _ = runtime_gate(signature_exit=signature_exit, smoke_exit=smoke_exit)
    assert process.returncode == (signature_exit or smoke_exit), process.stdout+process.stderr
    assert receipt['status'] == 'failed' and receipt['exit_code'] == process.returncode
    assert events == expected_events


def test_existing_extraction_and_old_smoke_cannot_make_a_retry_pass(runtime_gate):
    first, _, events, _ = runtime_gate()
    assert first.returncode == 0
    second, receipt, after, _ = runtime_gate()
    assert second.returncode != 0 and receipt['status'] == 'failed'
    assert after == events  # The second run never extracts, verifies or smokes.
