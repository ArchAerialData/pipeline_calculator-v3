"""Keep the approved Pyogrio wheel's GDAL/PROJ resources and native libraries."""
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_delvewheel_libs_directory,
    collect_dynamic_libs,
)

# Data directories stay package-relative; do not replace pyproj's distinct PROJ
# database with the one linked into GDAL. Wheel DLLs live beside the package on
# Windows, where ordinary collect_dynamic_libs/--collect-all cannot find them.
datas = collect_data_files("pyogrio", includes=["gdal_data/**", "proj_data/**"])
binaries = collect_dynamic_libs("pyogrio")
datas, binaries = collect_delvewheel_libs_directory("pyogrio", datas=datas, binaries=binaries)
hiddenimports = ["pyogrio._io", "pyogrio._ogr", "pyogrio._geometry", "pyogrio._err", "pyogrio._vsi"]
