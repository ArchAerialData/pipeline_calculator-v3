from __future__ import annotations

# Match tkinterdnd2.TkinterDnD._require: a bundle needs only the platform
# directory its Python process will load, including that directory's Tcl files.
import os
import platform
from pathlib import Path, PurePosixPath

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

system = platform.system()
machine = (os.environ.get('PROCESSOR_ARCHITECTURE', platform.machine())
           if system == 'Windows' else platform.machine())
platform_directories = {
    ('Darwin', 'arm64'): 'osx-arm64',
    ('Darwin', 'x86_64'): 'osx-x64',
    ('Linux', 'aarch64'): 'linux-arm64',
    ('Linux', 'x86_64'): 'linux-x64',
    ('Windows', 'ARM64'): 'win-arm64',
    ('Windows', 'AMD64'): 'win-x64',
    ('Windows', 'x86'): 'win-x86',
}
try:
    selected_platform = platform_directories[(system, machine)]
except KeyError:
    raise RuntimeError(f'tkinterdnd2 does not support {system}/{machine}; cannot package drag and drop.') from None

datas = []
selected_files = []
# On Unix, collect_data_files excludes .so files as Python extension candidates.
# tkdnd loads its shared library through Tcl, so collect those explicitly too.
resources = set(collect_data_files('tkinterdnd2') + collect_dynamic_libs('tkinterdnd2'))
for source, destination in sorted(resources):
    # Destinations use the build host's separators. Normalize them without
    # changing the original PyInstaller data tuple or unrelated package data.
    parts = PurePosixPath(str(destination).replace('\\', '/')).parts
    if parts[:2] == ('tkinterdnd2', 'tkdnd') and len(parts) > 2:
        if parts[2] != selected_platform:
            continue
        selected_files.append(Path(source))
    datas.append((source, destination))

library_suffix = {'Darwin': '.dylib', 'Linux': '.so', 'Windows': '.dll'}[system]
available = {path.name for path in selected_files if path.is_file()}
if (not {'pkgIndex.tcl', 'tkdnd.tcl'} <= available
        or not any(name.startswith('libtkdnd') and name.endswith(library_suffix) for name in available)):
    raise RuntimeError(f'tkinterdnd2 is missing the {selected_platform} Tcl files or native library; '
                       'reinstall the package before building.')

