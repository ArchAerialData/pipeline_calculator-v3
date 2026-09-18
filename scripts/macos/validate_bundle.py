"""Inventory the native dependencies of an app and stamp their minimum macOS.

Run before code signing. Header requirements are necessary compatibility limits,
not proof of successful execution on every macOS version above that limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import plistlib
import stat
import tempfile


CPU_TYPES = {'x86_64': 0x01000007, 'arm64': 0x0100000C}
MACHO_MAGICS = {bytes.fromhex(value) for value in (
    'feedface', 'cefaedfe', 'feedfacf', 'cffaedfe',
    'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca')}
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
MAX_BUNDLE_ENTRIES = 100_000


def version_tuple(value):
    value = int(value)
    if not 0 < value <= 0xFFFFFFFF or value >> 16 < 10:
        raise ValueError('Missing or invalid Mach-O minimum OS version')
    return value >> 16, (value >> 8) & 0xFF, value & 0xFF


def version_string(value):
    return '.'.join(str(component) for component in value)


def minimum_from_header(header):
    commands = [(load, data) for load, data, _ in header.commands
                if load.cmd in (LC_BUILD_VERSION, LC_VERSION_MIN_MACOSX)]
    if len(commands) != 1:
        raise ValueError('Expected exactly one macOS minimum-version load command')
    load, data = commands[0]
    if load.cmd == LC_BUILD_VERSION:
        if data.platform != 1:
            raise ValueError('Mach-O targets another Apple platform, not macOS')
        return version_tuple(data.minos)
    return version_tuple(data.version)


def read_minimum(path, architecture):
    # PyInstaller supplies this dependency on Darwin. Import lazily so the
    # filesystem and command policy tests can run on non-macOS hosts too.
    from macholib.MachO import MachO
    binary = MachO(str(path))
    matching = [header for header in binary.headers
                if header.header.cputype == CPU_TYPES[architecture]]
    if len(matching) != 1:
        raise ValueError(f'Expected exactly one {architecture} slice, found {len(matching)}')
    return minimum_from_header(matching[0])


def bundle_files(app):
    """Follow internal links once, refusing external or broken bundle links."""
    pending, seen, count = [app], set(), 0
    while pending:
        path = pending.pop()
        count += 1
        if count > MAX_BUNDLE_ENTRIES:
            raise ValueError('App bundle exceeds the validation entry limit')
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(app):
            raise ValueError(f'App bundle link escapes its root: {path}')
        if resolved in seen:
            continue
        seen.add(resolved)
        mode = resolved.stat().st_mode
        if stat.S_ISDIR(mode):
            pending.extend(sorted(resolved.iterdir(), reverse=True))
        elif stat.S_ISREG(mode):
            yield resolved
        else:
            raise ValueError(f'Unsupported special file in app bundle: {path}')


def atomic_write(path, content, *, mode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        if mode is not None:
            temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_bundle(app, architecture=None, *, output):
    app = Path(app).resolve(strict=True)
    architecture = architecture or platform.machine()
    if architecture not in CPU_TYPES:
        raise ValueError(f'Unsupported app architecture: {architecture}')
    if not app.is_dir():
        raise ValueError('Expected an app bundle directory')
    info_path = app / 'Contents/Info.plist'
    # Validate every link before reading or changing any bundle file.
    files = list(bundle_files(app))
    info_path = info_path.resolve(strict=True)
    with info_path.open('rb') as stream:
        info = plistlib.load(stream)
    if not isinstance(info, dict):
        raise ValueError('Info.plist must contain an application dictionary')
    executable_name = info.get('CFBundleExecutable')
    if not isinstance(executable_name, str) or not executable_name or Path(executable_name).name != executable_name:
        raise ValueError('Info.plist does not name a single bundle executable')
    executable = (app / 'Contents/MacOS' / executable_name).resolve(strict=True)
    if not executable.is_relative_to(app):
        raise ValueError('Bundle executable escapes the app')
    rows, versions, native_paths = [], [], set()
    for path in files:
        with path.open('rb') as stream:
            magic = stream.read(4)
        if magic not in MACHO_MAGICS:
            continue
        try:
            minimum = read_minimum(path, architecture)
        except Exception as error:
            raise ValueError(f'Cannot validate {path.relative_to(app)}: {error}') from error
        versions.append(minimum)
        native_paths.add(path)
        rows.append({'path': path.relative_to(app).as_posix(),
                     'architecture': architecture, 'minimum_macos': version_string(minimum)})
    if executable not in native_paths:
        raise ValueError('Bundle executable is not a validated Mach-O binary')
    required = version_string(max(versions))
    manifest = {'schema_version': 1, 'architecture': architecture,
                'required_minimum_macos': required, 'mach_o_file_count': len(rows),
                'binaries': sorted(rows, key=lambda row: row['path']),
                'compatibility_scope': 'Required by binary headers; runtime compatibility needs native OS testing.'}
    info['LSMinimumSystemVersion'] = required
    plist_bytes = plistlib.dumps(info)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode('utf-8')
    output = Path(output).resolve()
    if output.is_relative_to(app):
        raise ValueError('The validation manifest must be outside the app bundle')
    # All binaries must pass before the app is changed. Preserve plist access
    # permissions; stamping after signing would invalidate its signature.
    atomic_write(output, manifest_bytes)
    atomic_write(info_path, plist_bytes, mode=stat.S_IMODE(info_path.stat().st_mode))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('--architecture', choices=sorted(CPU_TYPES))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_bundle(args.app, args.architecture, output=args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f'App validation failed: {error}\n')
    print(f"Validated {result['mach_o_file_count']} {result['architecture']} binaries; "
          f"minimum macOS {result['required_minimum_macos']}")


if __name__ == '__main__':
    main()
