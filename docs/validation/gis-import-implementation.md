# GIS import implementation and validation

Implementation date: November 16, 2026. Based on source commit
`7d6e68314148f659798e7ffb75f287b3f2a6f7fe`; changes are local and uncommitted.

The follow-up [launcher and CI verification](gis-import-ci-readiness.md) records
the Bash alias check, updated Actions, additional macOS portability fixes, and
validation against a fresh Windows dependency environment.

## Delivered behavior

Browse or drop KML/KMZ, SHP, GPKG, GeoJSON, or a ZIP package directly into either
desktop entrypoint. ZIPs are inventoried and staged privately as needed; the user
does not extract files. Multiple usable layers require selection. Missing or
unusable CRS metadata requires an explicit source CRS assignment.

The shared [import package](../../src/pipeline_calculator/importing/__init__.py)
captures source bytes once, preserves independent multipart paths and source
identities, normalizes horizontal coordinates, and supplies fresh mutable records
to the existing analyzer. Reanalysis survives changed or removed originals.
Native GDAL reads run in a spawned process that can be terminated on cancellation.

Lines contribute mileage and actual points contribute pins. Polygons are
explicitly excluded from centerline mileage. Curves/surfaces are refused before
native approximation. Null geometry is reported without fabricating coordinates.
Source Z/M values do not affect horizontal mileage. The existing GRS80/US
survey-mile calculation and vertex-to-vertex geodesic convention remain unchanged.

Generic dataset/layer/feature identity and coordinate-operation provenance flow
through pipeline rows, point inventories, state fragments, JSON, XLSX, and KMZ.
KML folder columns and repair behavior remain compatible. GIS workbooks add
source columns and an Import Details sheet. Unselected/unsupported package
members are distinguished from invalid geometry in selected layers.

## Direct sample evidence

The supplied originals were preserved. Byte-identical fixture ZIPs and the
independent reference are described in the
[fixture notes](../../tests/fixtures/gis/customer_zip/README.md) and
[inspection JSON](gis-import-samples.json).

| Input | Usable line features | Paths | Vertices | Null records | Reference meters |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF_Lehman_Pipes.zip | 228 | 234 | 3,458 | 0 | 225,341.9297198959 |
| Verdun_Pipes.zip | 1,530 | 1,584 | 31,277 | 4 | 1,770,249.6445552604 |

Customer regression tests compare every source path against PyShp, every
non-null record against the saved length reference, total meters, physical null
record positions, counts, and original hashes. They also check loose-sidecar
parity and the actual desktop analysis/state routes. Verdun's null record
positions are 1514, 1515, 1520, and 1527 (zero-based); results are explicitly
incomplete for its selected source inventory.

The reference uses the same geodesic library family as the application, not an
independent geodesic algorithm. Independence here is in source reading, multipart
accounting, and per-record reference calculation. NAD83 operation EPSG:1188 is
explicitly recorded for these extents; its 4 m accuracy describes the coordinate
operation, not survey accuracy or a mileage error bound.

The full controller regression uses 1,000 m segmentation and a very large minimum
parallel length to keep import/state tests bounded and avoid generating overlap
corridor graphics. All original geometry and source mileage assertions are
unchanged. Existing overlap tests separately exercise ordinary analysis settings;
these tests do not claim a new default-settings overlap oracle for these samples.

## Verification

- Existing non-native regression suite: **1,001 passed, 14 skipped, 100 native
  tests deselected**, 362.16 seconds. Skips were five optional geography audit
  dependency checks and nine macOS Mach-O checks unavailable on this Windows host.
- GIS non-native regression suite: **94 passed across disjoint batches**: 79
  integration tests (165.66 seconds), 13 resource-limit tests, one startup-failure
  reporting test, and one shared-runtime packaging regression. Coverage includes
  both customer ZIPs, loose SHP parity, multiple GPKG layers, GeoJSON,
  points/collections, exact attributes, CRS conversion, source/state/export
  provenance, malformed inputs, snapshots, cancellation, and packaging hooks.
- Full native GUI suite: **108 passed, 5 skipped, 1,089 deselected**, 507.06
  seconds. The five optional-dependency skips also occur in the non-native run.
  GIS cases include drops of both customer ZIPs and retries after replacing the
  selected temporary original.
- Total distinct passing tests across those groups: **1,203**, with **14 distinct
  skips**. This is a combined result, not a single monolithic pytest invocation.
- Windows frozen development build: both modern and legacy interfaces **passed**
  the packaged smoke with `PROJ_NETWORK=OFF`. Each passed actual SHP, direct ZIP,
  GPKG layer selection, GeoJSON, projected-coordinate conversion, captured-source
  reuse, existing geography/repair/corridor probes, and 20 summary-return cycles
  without callback errors. The exact version, artifact SHA-256, and full reports
  are saved in [Windows package evidence](gis-import-windows-package.json).
- `git diff --check` passed; only the repository's usual LF-to-CRLF notices were
  emitted.

### Defects found and fixed during validation

- Deleted SHP DBF rows changed GDAL's batch cursor semantics; advancing by the
  number of returned active rows repeated a physical feature. The cursor now
  advances past the last physical FID, with active-record reconciliation and
  duplicate-FID rejection. A 260-record fixture with one deleted row produces
  exactly 259 distinct features.
- A valid geographic CRS using grads was rejected by a degrees-only coordinate
  range check. Source angular-unit conversion now controls the check; the
  transformed output is still validated in longitude/latitude degrees.
- Nonempty but unusable CRS text triggered a decision without an editable CRS
  field. The modal now uses the explicit missing-CRS decision IDs.
- Nullable integer attributes could be rounded by a reader's float conversion.
  GPKG attributes are recovered from the captured SQLite source, integral DBF
  fields retain exact original values, and GeoJSON retains original properties.
- Generic and nested GPKG geometries are audited before conversion, including
  source SRID agreement. Nested ZIP/KMZ and mixed KML omissions are disclosed.
- The first frozen build placed a shared MSVC runtime only in `pyogrio.libs`;
  legacy startup needed its copy in `pandas.libs` before Pyogrio was imported.
  The supplemental Pandas hook now preserves that wheel's runtime directory as
  well. The rebuilt executable passed both interfaces. Smoke initialization now
  reports startup import failures as JSON rather than leaving a windowed error
  dialog to time out.

## Dependencies and distribution

Runtime pins: `pyogrio==0.13.0`, `pyproj==3.7.2`; PyInstaller minimum is 5.6.
The tested Windows wheel reports GDAL 3.12.4. GeoPandas, Fiona, and PyArrow are
not required. The reader uses public Pyogrio APIs; bounded format audits check
original semantics rather than substituting another production decoder.

The [Pyogrio hook](../../scripts/pyinstaller_hooks/hook-pyogrio.py) retains wheel
native libraries and package-relative GDAL/PROJ resources. The Windows and macOS
build scripts invoke that hook. A supplemental
[Pandas hook](../../scripts/pyinstaller_hooks/hook-pandas._libs.py) preserves its
own Windows runtime DLLs without replacing the standard Pandas hook.
Entry points dispatch multiprocessing children
before GUI startup. The offline packaged smoke requires actual SHP/ZIP/GPKG/
GeoJSON reads, projected conversion, layer selection, and captured-source reuse.

macOS build integration is implemented, but actual Apple Silicon builds,
macOS 15/26/27 execution, signing, and notarization cannot be verified on this
Windows host. Those remain release gates; no macOS distribution is claimed.

Input caps (256 MiB captured/expanded, 10,000 archive entries, 512 layers,
250,000 features, 2,000,000 vertices, plus independent attribute/IPC/depth bounds)
are safeguards, not near-capacity performance guarantees. No broad capacity or
large-customer-corpus benchmark was performed. Standalone API scripts require
the normal `if __name__ == '__main__':` guard for process spawning.

## Reproduction

```powershell
.venv\Scripts\python.exe -m pytest tests -q -m 'not native_gui' '--ignore-glob=tests/test_gis_*.py'
$gisTests = @(Get-ChildItem -LiteralPath tests -Filter 'test_gis_*.py' | ForEach-Object { $_.FullName })
.venv\Scripts\python.exe -m pytest @gisTests -q -m 'not native_gui'
.venv\Scripts\python.exe -m pytest tests -q -m native_gui
```

The frozen validation uses `scripts/windows/build_exe.ps1 -OutputRoot` to place
builds under `.validation-output/gis-build`, preserving existing `build/` and
`dist/` artifacts. The smoke checker runs both modern and legacy interfaces with
network projection downloads disabled. Artifacts are local development previews,
not a published or signed release.
