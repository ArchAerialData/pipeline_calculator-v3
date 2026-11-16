"""Preserve pandas' own Windows wheel DLL directory, independently of GDAL.

Some pandas and Pyogrio wheels carry the same hashed MSVC runtime filename.
Dependency scanning alone can resolve pandas' copy to the already-collected
Pyogrio directory. That directory is not registered when pandas imports first.
Collect the wheel's declared pandas.libs directory explicitly, without replacing
PyInstaller's standard top-level pandas hook or relying on import order.
"""
from PyInstaller.utils.hooks import collect_delvewheel_libs_directory

datas, binaries = collect_delvewheel_libs_directory("pandas")
