"""Exercise real packaging helpers with isolated macOS command boundaries.

These tests prove failure propagation and distribution command contracts. They
do not substitute for native codesign, ditto, Gatekeeper, or notarization checks.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def bash_path(path):
    value = path.resolve().as_posix()
    return '/' + value[0].lower() + value[2:] if sys.platform == 'win32' else value


def native_path(value):
    """Compare physical paths across MSYS and macOS /var symlink spellings."""
    if sys.platform == 'win32' and len(value) >= 3 and value[0] == '/' and value[2] == '/':
        value = value[1] + ':' + value[2:]
    return Path(value).resolve()


def executable(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + body, newline='\n')
    path.chmod(0o755)


@pytest.fixture
def distribution(tmp_path):
    bash = shutil.which('bash')
    if sys.platform == 'win32':
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        bash = str(git_bash) if git_bash.exists() else None
    if not bash:
        pytest.skip('Bash is required to execute the packaging helper boundary test')
    repo = tmp_path / 'distribution workspace'
    tools = repo / 'tools'
    tools.mkdir(parents=True)
    scratch = repo / 'temporary signing materials'
    scratch.mkdir()
    app = repo / 'dist/Pipeline_Calculator_v5.app'
    executable(app / 'Contents/MacOS/Pipeline Calculator', 'exit 0\n')
    (app / 'Contents/Info.plist').write_text('Version is supplied by the isolated PlistBuddy boundary.')
    for relative in ('scripts/ci/macos_sign_and_package.sh', 'scripts/macos/package_dmg.sh'):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the helper body, replacing only the absolute macOS tool path.
        text = (ROOT / relative).read_text().replace('/usr/libexec/PlistBuddy', '"${TEST_TOOLS}/PlistBuddy"')
        destination.write_text(text, newline='\n')
    executable(tools / 'PlistBuddy', "printf '%s\\n' '4.26-dev.embedded'\n")
    executable(tools / 'python3', "printf '%s\\n' 'fake-keychain-password'\n")
    executable(tools / 'mktemp', '''
# Apple mktemp's implicit template prefers confstr(_CS_DARWIN_USER_TEMP_DIR)
# over TMPDIR. Supply an explicit path while retaining native atomic creation.
[[ "$#" -eq 1 && "$1" == '-d' ]] || exit 96
exec /usr/bin/mktemp -d "$TMPDIR/signing.XXXXXXXXXX"
''')
    executable(tools / 'security', '''
printf 'security %s\n' "$1" >> "$TEST_LOG"
if [[ "$1" == 'find-identity' ]]; then
  printf '%s\n' ' 1) 123456 "Developer ID Application: Boundary Test (TEST)"'
fi
''')
    executable(tools / 'codesign', '''
printf 'codesign %s\n' "$1" >> "$TEST_LOG"
if [[ "$1" == '--verify' ]]; then exit "$VERIFY_EXIT"; fi
''')
    executable(repo / '.venv/bin/python', '''
printf 'smoke\n' >> "$TEST_LOG"
printf '%s\n' "$@" > "$TEST_REPO/smoke-arguments.txt"
exit "$SMOKE_EXIT"
''')
    executable(tools / 'hdiutil', '''
printf 'hdiutil\n' >> "$TEST_LOG"
printf '%s\n' "$@" > "$TEST_REPO/hdiutil-arguments.txt"
printf 'fake DMG\n' > "${!#}"
''')
    executable(tools / 'ditto', '''
printf 'ditto\n' >> "$TEST_LOG"
printf '%s\n' "$@" > "$TEST_REPO/ditto-arguments.txt"
printf 'fake app ZIP\n' > "${!#}"
''')
    executable(tools / 'ln', '''
# A native /Applications link does not exist on the Windows test host.
printf '%s\n' "$@" > "$TEST_REPO/link-arguments.txt"
touch "${!#}"
''')
    executable(tools / 'rm', '''
# Every recursive cleanup is confined to the disposable test repository.
# Use the real interpreter, not our python3 signing-password stub. Physical
# resolution accepts /var == /private/var but still rejects symlink/.. escapes,
# the workspace itself, and siblings that merely share its string prefix.
"$TEST_PYTHON" - "$TEST_NATIVE_REPO" "$@" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    if value.startswith('-'):
        continue
    target = Path(value).resolve()
    if target == root or not target.is_relative_to(root):
        print(f'Refusing cleanup outside fixture: supplied={value!r}, resolved={target}, fixture={root}', file=sys.stderr)
        sys.exit(97)
PY
exec /bin/rm "$@"
''')
    env = dict(os.environ,
               TEST_REPO=bash_path(repo), TEST_TOOLS=bash_path(tools),
               TEST_PYTHON=bash_path(Path(sys.executable)), TEST_NATIVE_REPO=str(repo.resolve()),
               TEST_LOG=bash_path(repo / 'events.log'), TMPDIR=bash_path(scratch),
               MACOS_CERT_P12='dGVzdA==', MACOS_CERT_PASSWORD='fake-certificate-password',
               VERIFY_EXIT='0', SMOKE_EXIT='0')
    env.pop('ARTIFACT_ARCH', None)

    def run(*, verify_exit=0, smoke_exit=0, arch=None):
        current = dict(env, VERIFY_EXIT=str(verify_exit), SMOKE_EXIT=str(smoke_exit))
        if arch is not None:
            current['ARTIFACT_ARCH'] = arch
        script = repo / 'scripts/ci/macos_sign_and_package.sh'
        current['TEST_SCRIPT'] = bash_path(script)
        process = subprocess.run(
            [bash, '-c', 'export PATH="$TEST_TOOLS:$PATH"; exec bash "$TEST_SCRIPT"'],
            env=current, cwd=repo, capture_output=True, text=True, timeout=25)
        events = (repo / 'events.log').read_text().splitlines()
        return process, events

    return repo, scratch, app, run


@pytest.mark.parametrize('verify_exit,smoke_exit,expected', [(23, 0, 23), (0, 24, 24)])
def test_signed_verification_or_smoke_failure_never_packages(distribution, verify_exit, smoke_exit, expected):
    repo, scratch, _, run = distribution
    process, events = run(verify_exit=verify_exit, smoke_exit=smoke_exit, arch='arm64')
    assert process.returncode == expected, process.stdout + process.stderr
    assert 'codesign --force' in events and 'codesign --verify' in events
    assert ('smoke' in events) == (verify_exit == 0)
    assert not any(event in events for event in ('hdiutil', 'ditto'))
    assert not list((repo / 'dist').glob('*.dmg'))
    assert not list((repo / 'dist').glob('*.app.zip'))
    assert not (repo / 'dist/dmg_staging').exists()
    assert 'security delete-keychain' in events and not list(scratch.iterdir())


@pytest.mark.parametrize('arch', ['arm64', 'x86_64', None])
def test_distribution_names_and_preserving_app_archive_contract(distribution, arch):
    repo, scratch, app, run = distribution
    process, events = run(arch=arch)
    assert process.returncode == 0, process.stdout + process.stderr
    assert events.index('codesign --verify') < events.index('smoke') < events.index('hdiutil') < events.index('ditto')
    suffix = '_' + arch if arch else ''
    stem = f'Pipeline_Calculator_v4.26-dev.embedded{suffix}'
    assert (repo / 'dist' / (stem + '.dmg')).is_file()
    assert (repo / 'dist' / (stem + '.app.zip')).is_file()
    ditto = (repo / 'ditto-arguments.txt').read_text().splitlines()
    assert ditto[:4] == ['-c', '-k', '--sequesterRsrc', '--keepParent']
    assert [native_path(value) for value in ditto[4:]] == [app.resolve(),
                        (repo / 'dist' / (stem + '.app.zip')).resolve()]
    smoke = (repo / 'smoke-arguments.txt').read_text().splitlines()
    assert smoke[smoke.index('--expected-version') + 1] == '4.26-dev.embedded'
    assert native_path(smoke[1]) == app.resolve()
    link = (repo / 'link-arguments.txt').read_text().splitlines()
    assert link[:2] == ['-s', '/Applications']
    assert not (repo / 'dist/dmg_staging').exists()
    assert app.is_dir() and not list(scratch.iterdir())
