from __future__ import annotations

# Ensure tkinterdnd2's bundled tkdnd libraries get included in PyInstaller builds.

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("tkinterdnd2")

