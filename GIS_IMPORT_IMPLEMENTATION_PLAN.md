# GIS import and direct ZIP drag-and-drop

Status: **Implemented; validation and platform release status are recorded in the
[implementation report](docs/validation/gis-import-implementation.md).**
Investigated November 16, 2026, against commit
`7d6e68314148f659798e7ffb75f287b3f2a6f7fe`.

The design and original sample-investigation evidence below are retained as the
implementation specification. The linked report distinguishes completed work,
automated verification, and remaining platform release gates.

## 1. Recommended scope

Add a shared GIS import service that feeds the existing analysis engine. The
first release should accept **SHP, GPKG, GeoJSON, and ZIP packages containing
those datasets**, alongside existing KML/KMZ. A user drops either supplied ZIP
directly onto the app; the app discovers its shapefile components, reads its
projection, and analyzes its lines without any manual extraction.

Keep geometry meaning consistent across formats: lines contribute mileage,
points contribute pin counts, and polygons are identified and excluded from
centerline mileage. Adding polygon area/perimeter analysis is a separate feature,
as already deferred in [the geometry-counts plan](GEOMETRY_COUNTS_FEATURE_PLAN.md).

This change covers discovery, layer choices, coordinates, provenance, UI, exports,
tests, and packaged delivery. Merely adding file extensions to the picker is not
sufficient. The existing KML parser and verified repair workflow remain a
specialized adapter behind the shared service.

### User-facing behavior

1. Browse or drop **one input file/container**. A ZIP is one input even when it
   holds multiple datasets. Ordinary filesystem folder drops and combining
   several outer files in one gesture are deferred; do not silently take the
   first dropped file.
2. Inspect on a worker. If exactly one eligible vector layer exists and its CRS
   is usable, proceed directly. The two supplied ZIPs should take this path.
3. If several eligible layers exist, show **Choose layers to analyze** with
   dataset/member path, layer name, geometry family, CRS, and feature count when
   available. Require an explicit selection; offer Select all. Do not guess the
   primary pipeline layer or automatically merge apparent duplicates.
4. Show unsupported layers and their reason. A selected mixed layer can contribute
   lines and points independently. Keep the selected scope visible in results.
5. For missing CRS, offer a compact **Specify source coordinate system** decision
   accepting an EPSG code or WKT, or let the user choose another file. Explain
   that this assigns the source CRS. Validate the choice before transforming;
   never infer it from coordinate ranges. No such question is needed for these
   samples because both include valid `.prj` files.
6. Continue to normal results, state breakdown, overlap analysis, and exports.
   Report omissions where they affect completeness. Retry/reanalysis uses the
   captured source and previous layer/CRS choices.

## 2. Directly verified sample evidence

Both inputs are ordinary ZIPs with one matching shapefile set at archive root.
Each contains `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`, `.sbn`, `.sbx`, and
`.shp.xml`. ZIP CRC checks pass. Their `.cpg` declares UTF-8 and their `.prj`
resolves to **NAD83 / EPSG:4269**. Neither is a GeoPackage example.

| Inspection | RF_Lehman_Pipes.zip | Verdun_Pipes.zip |
| --- | ---: | ---: |
| Original ZIP bytes | 91,284 | 508,361 |
| Declared shape type | PolyLineZ | PolyLineZ |
| Source shape records | 228 | 1,534 |
| Non-null line features | 228 | 1,530 |
| Separate line paths | 234 | 1,584 |
| XY vertices | 3,458 | 31,277 |
| Multipart features | 6 | 32 |
| Null geometries | 0 | 4 |
| Paths shorter than two vertices | 0 | 0 |
| Non-finite XY positions | 0 | 0 |

These are structurally useful regression fixtures. **Verdun must not be reported
as a complete inventory of 1,534 measured pipelines:** four records provide no
geometry. Analyze its 1,530 usable features, retain all 1,584 paths, and report the
four excluded source records with incomplete-input status. Do not infer missing
geometry from attributes. RF has a `LENGTH_MI` attribute; measured mileage must
still come from coordinates. Both have `PLINE_ID` and `GlobalID`, which must not
be assumed unique without validation.

The independent reference uses the existing GRS80 geodesic/US survey-mile
convention after explicit NAD83-to-WGS84 operation EPSG:1188 for these mainland
extents. Its coordinate operation is a no-op with published **4 m operation
accuracy**, not proof of exact datum equality or of the source's positional
accuracy. This matches the explicit mainland operation already used by
[state-boundary preparation](scripts/data/prepare_state_boundaries.py#L26).

Evidence and reproduction:

- [Read-only inspection script](scripts/validation/inspect_gis_archives.py).
- [Machine-readable inventory and references](docs/validation/gis-import-samples.json).
- [Fixture copies and usage notes](tests/fixtures/gis/customer_zip/README.md).

The report records hashes, member sizes, feature/part counts, null-record
identities, dimensions, CRS, ordered XY digests, and reference lengths. Original
files stay in Downloads unchanged. Copies remain byte-identical ZIPs, so future
tests exercise direct archive import. XML metadata is treated as opaque source
data; document contents are not task instructions.

This investigation verifies archive readability with PyShp and coordinate
metadata with pyproj. It does **not** verify the proposed production backend,
GUI import, GPKG compatibility, overlap results, or frozen distribution support.

## 3. Format and geometry contract

### Formats in the first release

| Input | Required behavior |
| --- | --- |
| `.kml`, `.kmz` | Existing parsing, local links, tracks, folders, repair, and snapshot semantics remain intact. |
| `.shp` | Capture matching `.shp/.shx/.dbf` siblings as one dataset; use `.prj` and `.cpg` when supplied. Require CRS resolution before analysis. |
| `.gpkg` | Discover vector feature layers; independently resolve each layer's CRS and identity. Show raster/nonspatial contents as unavailable for pipeline analysis. |
| `.geojson`, `.json` | Validate GeoJSON content; a `.json` suffix alone does not establish GIS input. Accept RFC 7946 coordinates in longitude/latitude order. Reject conflicting legacy CRS declarations with actionable re-export guidance. |
| `.zip` | Discover complete SHP sets, GPKG files, and GeoJSON files at root or in nested directories; use the same reader contracts as loose input. |

A KML-only ZIP may use the existing KMZ package-selection/link rules once its
content is classified as KML packaging; do not treat every linked KML document
as another independent dataset. For the first release, mixed generic ZIPs that
also contain KML show that KML package as unsupported and direct the user to
open it separately as KML/KMZ. Nested ZIP/KMZ archives are not recursively opened.
Record ignored nested packages in discovery so the user does not assume they
were analyzed. Existing direct KMZ support is unchanged.

GPX, FlatGeobuf, zipped FileGDB, GeoParquet, CAD, raster analysis, remote services,
and new GIS export formats are follow-up adapters. Do not advertise every driver
installed by GDAL as supported. The initial format list is a tested allowlist.

### Geometry normalization

| Encoded geometry | Normalized result |
| --- | --- |
| LineString / SHP PolyLine | One source feature becomes one pipeline. |
| MultiLineString / multipart PolyLine | One pipeline with multiple `coordinate_paths`; never insert a connector between parts. |
| Point | One pin with its actual XY location retained in the normalized model. |
| MultiPoint | One pin per component, with parent feature identity and component index; coincident pins remain distinct. |
| GeometryCollection | Recursively classify bounded child geometries; collect lines and points independently, preserving component paths. |
| Polygon / MultiPolygon / ring | Count exclusions by layer/type for diagnostics; no mileage, no pins from vertices, no implicit centerline extraction. |
| Closed or self-crossing line | Remains a line; appearance does not change its encoded type. |
| Z / M variants | Analyze horizontal XY only, disclose unused Z/M, preserve source bytes and dimension provenance. No 3D mileage or vertical datum claim. |
| Null / empty geometry | Retain source identity in omission diagnostics; no fabricated coordinates. Selected-input completeness is false. |
| Malformed line / point | Reject that feature atomically, never reconnect across invalid vertices; return partial results only with explicit omissions. |
| Curves, TIN, surfaces, multipatch | Unsupported in this release. Do not silently linearize, repair, simplify, snap, union, or deduplicate. |

Layer declarations are discovery hints; individual features still need checking.
GDAL documents that one Shapefile line layer can contain both single and multipart
features. [GDAL Shapefile driver](https://gdal.org/en/stable/drivers/vector/shapefile.html).

Intentional polygon exclusions and deliberately unselected layers do not mean
the selected supported geometry is damaged. Null, malformed, unsupported curve,
or unreadable geometry **inside a selected layer** does mean its inventory is
incomplete. Polygon-only input gives a clear no-supported-features result;
points-only input remains a valid zero-mile analysis. Preserve the existing
`analysis_complete` contract and add structured coverage counts by reason.

### Coordinates and measurement

- Normalize source XY to **EPSG:4326 with longitude first**, using `always_xy=True`
  and checked transformations. Distinguish a CRS from the GRS80 ellipsoid used by
  the calculator. Projected meters and US/international feet are resolved by CRS,
  not suffixes, attribute names, or value magnitude.
- Read source CRS from `.prj`, each GPKG layer, or the GeoJSON format contract.
  Save full CRS WKT as well as authority identifiers when available. A valid WKT
  need not have an EPSG match. Missing/undefined GPKG SRS and unparseable `.prj`
  require a recorded user assignment, not a silent WGS84 default.
- Select an offline operation whose area covers the data; retain operation ID or
  pipeline, area, stated accuracy, grids, PROJ/database version, and any user CRS
  assignment. Check finite output/ranges and transformation failures. Data
  spanning distinct operation areas needs per-part operations or an actionable
  unsupported-transform result, not one arbitrarily chosen regional operation.
- Do not download grids during imports. If the required operation is unavailable,
  report the missing resource and stop that import. A documented, available
  lower-accuracy operation may be used only under an explicit policy recorded in
  provenance; no automatic ballpark fallback. EPSG:1188 is the explicit policy for
  these sample extents. Reanalysis reuses the selected operation.
- Preserve original vertex order and part boundaries. **Version 1 measures
  geodesic edges between transformed source vertices**, matching current KML
  analysis. Do not add hidden densification. For sparse projected data this is
  different from transforming every point of a straight projected edge; document
  that measurement convention and test a deliberately long projected edge.
  Source-space densification would be a separately versioned measurement option.
- Keep GRS80 distance and `1609.347218694` meters per US survey mile unchanged.
  Source length fields are descriptive attributes only. Z/M never change XY length.

Sources: [current length calculation](src/pipeline_calculator/core/analyzer.py#L61),
[measurement constants](src/pipeline_calculator/core/constants.py#L12),
[pyproj operation selection and areas](https://pyproj4.github.io/pyproj/stable/advanced_examples.html),
[GeoJSON coordinate specification](https://www.rfc-editor.org/rfc/rfc7946).

## 4. Maintainable import architecture

Add focused modules under `src/pipeline_calculator/importing/`; avoid growing
`kml_kmz.py` into a universal parser.

| Component | Responsibility |
| --- | --- |
| `formats.py` | Tested extension/content registry, browse filters, driver allowlist, capabilities. |
| `models.py` | Dataset/layer descriptors, selections, normalized feature identity, diagnostics, provenance. |
| `session.py` | Captured source ownership, decisions, fresh parse copies, reference counting, cleanup. |
| `archive.py` | Bounded inventory, component pairing, selective private materialization. |
| `coordinates.py` | CRS validation, operation selection, checked XY conversion and provenance. |
| `normalize.py` | Geometry classification, multipart preservation, pin expansion, coverage accounting. |
| `readers/kml.py` | Bridge to existing `prepare_source`/`SourceSession`; KML repair stays isolated. |
| `readers/ogr.py` | Allowlisted SHP/GPKG/GeoJSON discovery and decoding. |

Use a small explicit reader protocol (`discover`, `read_selected`) and one generic
session interface (`retain`, `release`, `fresh_parse`, provenance, capabilities).
The session contains the original identity, content hashes, captured components,
discovered layers, chosen layers, CRS decisions, and normalized immutable data.
Every analysis obtains a fresh mutable copy because overlap caches mutate records.
Only KML sessions expose repair approval and repaired-copy export capabilities.

Implement `prepare_import` as the shared entry point for both the desktop worker
and direct `PipelineAnalyzer.analyze_complete` path. Keep old parser functions as
KML-specific compatibility APIs; do not change their meaning silently. Headless
callers must be able to supply layer/CRS choices. When choices are needed and
absent, return/raise typed decision requirements rather than opening UI or taking
the first layer. `analyze_parsed` remains the core bridge.

Evidence for this split:
[direct analyzer import](src/pipeline_calculator/core/analyzer.py#L24),
[desktop preparation](src/pipeline_calculator/gui/controllers/analysis_controller.py#L40),
[existing source session](src/pipeline_calculator/parsers/source.py#L392), and
[fresh-record contract](src/pipeline_calculator/core/analyzer.py#L197).

### Identity, attributes, and compatibility

- Internal identity is `(source digest, dataset member, layer, source FID/record
  ordinal, component path)`. Use stable distinct IDs even when client IDs/names
  repeat. A multipart line stays one pipeline; its parts get separate identities.
- Retain source FID, client identifiers, original field names/types, and bounded
  attributes separately. Handle missing/NaN/date/64-bit integer values explicitly;
  never use a lossy float-coerced identifier as the sole record identity.
- Deterministic display-name preference: nonblank `Name`/`NAME`, then `PLINE_ID`,
  then `<layer> feature <source ID>`. Do not merge on display names. Preserve
  `GlobalID`, `OBJECTID`, `PLINE_ID`, and layer identifiers independently.
- Add generic `source_dataset`, `source_layer`, `source_feature_id`, and component
  provenance through state splits, tables, XLSX, JSON, and KMZ output metadata.
  Keep KML folder provenance and existing JSON keys for compatibility; do not fill
  `source_kml` with a shapefile path or pretend a GIS layer is a KML folder.
- Introduce a generic provenance-copy helper used by analysis and export.
  The current [folder helper](src/pipeline_calculator/core/placemarks.py#L5)
  copies only KML-oriented fields and otherwise would drop new source metadata.
- Group GIS pins by layer, KML pins by folder, and label each honestly. Retain
  point coordinates in the normalized model although current point UI counts only.
  Add generic workbook columns alongside existing KML-specific ones; version new
  JSON import metadata. Preserve old-result rendering and export compatibility.

### Reader dependency decision and feasibility gate

Recommended backend: **one pinned Pyogrio/GDAL stack**, isolated behind the reader
adapter, using NumPy/WKB plus the existing Shapely/pyproj stack. Avoid requiring
GeoPandas or PyArrow just to import geometry. Keep PyShp as an independent
development-only shapefile oracle rather than maintaining two production paths.
Do not implement a general Shapefile or GeoPackage decoder from scratch.

Pyogrio distributes wheels with GDAL and common vector drivers; its raw reader
returns geometry/attribute arrays. Exact wheel, Python 3.11, NumPy 1.26.4, and
frozen-app compatibility still need a measured spike; no version was installed
or approved by this investigation.
[Installation](https://pyogrio.readthedocs.io/en/latest/install.html),
[raw reader source](https://github.com/geopandas/pyogrio/blob/main/pyogrio/raw.py).

**Critical gate:** Pyogrio documents automatic approximation of curves/surfaces
and reduction of measured geometries. Metadata checks after decoding are not
sufficient proof that source geometry was preserved. The spike must demonstrate
detection of unsupported source geometry **before** conversion, including curves
inside generic/collection GPKG layers. Inspect declared extensions/types and use
a bounded original-geometry type audit where needed. Any necessary GPKG header/
type audit is validation only; GDAL remains the decoder. Reject layers whose
original geometry semantics cannot be established. If this requires private
Pyogrio internals or cannot reliably detect coercion, reject that backend choice
and evaluate a public OGR feature-level reader before proceeding. Do not ship
silent approximation to meet a format checkbox.
[Pyogrio limitations](https://pyogrio.readthedocs.io/en/latest/known_issues.html),
[GeoPackage geometry encoding](https://www.geopackage.org/spec140/index.html).

Also verify raw nullable integers/FIDs, DBF encoding, deleted-record accounting,
and XML sidecar behavior. Stage only the supported sidecars for SHP import;
`.shp.xml` stays opaque so backend upgrades cannot unexpectedly reinterpret its
field aliases. Read all fields needed for provenance, not arbitrary XML content.
Pin the approved versions and record actual GDAL/PROJ driver versions in smoke
reports. Do not ship a second GDAL distribution through Fiona as a fallback.

## 5. ZIP handling and source lifetime

ZIP is a container, not a geometry format. Separate generic archive validation
from KMZ primary-document selection currently coupled in
[source.py](src/pipeline_calculator/parsers/source.py#L194).

1. Capture the outer input once with a hash and before/after stability checks.
   For loose SHP, capture all required sidecars as one logical snapshot and
   verify the whole set stayed unchanged across capture. Reject missing `.shx`
   or `.dbf` with exact required names; index reconstruction/geometry-only
   recovery is outside this first release.
2. Inventory members before opening any vector driver. Match SHP components by
   parent directory and basename, case-insensitively only when unambiguous. Keep
   distinct same-name datasets in separate subdirectories. Ignore harmless
   `__MACOSX`, `.DS_Store`, and unrelated documents with an inventory count.
3. Reject unsafe paths, traversal, absolute/drive paths, symlinks/special files,
   case/Unicode-normalization collisions, encrypted members, corrupt CRCs, and
   truncated content. Enforce limits on actual decompressed bytes, not just ZIP
   size headers. Compression ratio can support diagnostics but must not be the
   only limit.
4. Initial outer and total expanded byte limits: **256 MiB**, matching current
   sessions; entry limit **10,000**. Bound feature count, vertices, collection
   depth, attribute bytes, and retained diagnostics independently during decode.
   Set those additional numeric caps from the feasibility memory/cancellation
   measurements before implementation acceptance; do not present unknown limits
   as validated capacity. Limit failures never masquerade as complete input.
5. Materialize validated candidate datasets and necessary sidecars into an
   app-owned private workspace when discovery requires seekable access. In
   particular, a GPKG inside a ZIP must be available before its layers can be
   listed and selected. Bound this discovery staging by the same byte limits;
   decode analysis geometry only from selected layers. Use generated staging
   names plus an original-path manifest, never archive paths as filesystem
   destinations. Do not use unrestricted `extractall`. Release unused staging
   after selection while retaining the captured source for later choices.
   User-facing paths always refer to the original dataset, not staging names.
6. A private staged GPKG gives SQLite seekable access. Detect live `-wal`/journal
   sidecars and source mutation; ask for a closed/exported consistent dataset
   instead of copying only the main file and losing committed changes. Readers
   open read-only, never update customer files or enable SQLite extension loading.
7. Session lifetime owns staging. Keep it alive for active workers and retained
   results, including layer decisions, cancellation races, reanalysis, and
   exports. Delete only after the last owner releases it; clean abandoned
   app-owned workspaces safely on a later launch.

This supplies the requested no-manual-unzip workflow while keeping archive and
SQLite behavior deterministic. GDAL also offers direct `/vsizip/` access, but it
does not replace application validation, limits, or snapshot ownership. Private
materialization is the initial design; direct virtual reads are a later measured
optimization. [GDAL virtual filesystem documentation](https://gdal.org/en/stable/user/virtual_file_systems.html).

Keep reads local and driver-allowlisted. Dataset attributes, URLs, external
references, SQL-looking values, and documentation inside archives are data, not
instructions or requests to fetch resources. Allowlisted formats cannot enable
unrelated remote/VRT drivers via a misleading filename.

## 6. Desktop integration and diagnostics

Use a shared format registry for modern and legacy browse dialogs, shared drop
validation, retry/choose-another-file, and supported-format help. Preserve Tcl
`splitlist` handling of spaces, braces, and Unicode paths. Existing duplicated
filters are in [main_window.py](src/pipeline_calculator/gui/main_window.py#L111)
and [pipeline_calculator_v3.py](src/pipeline_calculator_v3.py#L300); suffix gating
is in [file_select_page.py](src/pipeline_calculator/gui/pages/file_select_page.py#L116).

Add explicit controller states for `discovering`, `layer_selection_required`,
`crs_required`, `reading`, and `normalizing`, alongside existing repair/analysis
states. Typed `ImportFailure` must not be caught as KML repair merely because it
contains findings; the current [repair handler](src/pipeline_calculator/gui/repair_ui.py#L265)
uses that pattern. All decisions are main-thread UI; all I/O/decoding stays off it.

Use counted progress stages when totals are known, indeterminate stages otherwise.
Check cancellation between archive chunks, layers, batches, and transformations.
Because native bulk reads may not yield Python cancellation checkpoints, run
new vector discovery/decode in a dedicated child process if the spike cannot
demonstrate bounded cancellation. The release requirement is cancellation within
two seconds on supported-capacity fixtures; enforce it with process termination
after a grace period when needed. Parent retains source ownership until the child
exits. Use bounded/versioned IPC; no evaluation of dataset values. Exercise this
helper through both frozen entrypoints, not just a source interpreter.

Keep UI findings actionable and grouped. Example for Verdun:
**“Imported 1,530 line features (1,584 paths). Four source records have no geometry
and are excluded from mileage.”** Provide record details in Diagnostics and
export metadata; do not show a modal per missing feature. On these Z sources,
state **“Horizontal length; elevation/measure values are not used.”**

Changing the input or canceling a layer/CRS decision must not publish stale
results. Reanalysis retains scope and coordinate decisions. Allow changing layer
selection through an explicit import action, not through ordinary parameter
adjustment. Results and exports identify the selected layers and omitted counts.
Update “KML documents read” and source labels without losing KML-specific detail.

## 7. Regression and acceptance plan

### Customer ZIP gates

The copied ZIPs are immutable evidence, not outputs to regenerate from the new
reader. Pin both original SHA-256 values recorded in the inspection report.

- Direct ZIP import yields the exact feature/path/vertex/null counts in section 2.
  Compare ordered XY geometry against PyShp with explicit feature/part boundaries;
  a plausible grand-total length alone cannot prove correctness.
- Compare per-feature and total lengths to the independent reference. For the
  fixed sample no-op transform use `abs_tol=0.001 m, rel_tol=1e-10`; exact counts
  and path structure must match independently of floating-point tolerances.
- RF is complete for supported source geometry; Verdun reports exactly four null
  records and incomplete selected-input inventory while analyzing all valid lines.
- ZIP and loose sidecars staged from that ZIP produce equivalent normalized
  results. Snapshot mutation/removal after capture cannot change reanalysis.
- Exercise both ZIPs through the real desktop worker with state breakdown off/on.
  Preserve existing analysis workload warning/skip semantics if overlap limits
  are reached; do not mistake a heavy-workload policy for import failure.

### Synthetic fixtures and tests to add during implementation

| Area / proposed test module | Required cases |
| --- | --- |
| `test_gis_formats.py` | SHP/GPKG/GeoJSON and ZIP parity; unsupported/mislabeled content; registry filters and driver allowlist. |
| `test_gis_geometry.py` | Point/MultiPoint, mixed collections, multipart gaps, polygons/holes, closed/crossing lines, null/empty, Z/M, repeated IDs, invalid vertex, recursion depth, unsupported curves before conversion. |
| `test_gis_crs.py` | EPSG:4326 axis order; NAD83; projected meters and feet; valid non-EPSG WKT; missing/undefined/invalid CRS; user assignment; unavailable grids; area mismatch; finite output; sparse projected edge convention. |
| `test_gis_archives.py` | Nested directories, several shapefile sets, multiple GPKGs/layers, missing sidecars, uppercase/Unicode names, collisions, traversal, symlink, encrypted/corrupt/truncated ZIP, declared-vs-actual limits, unrelated files, unsupported nested archive. |
| `test_gis_sessions.py` | Frozen source/sidecars; mutation during capture; repeated fresh parses; cancellation/cleanup; GPKG WAL/journal refusal; layer/CRS choice lifetime and stale results. |
| `test_gis_import_ui.py` | Both GUI entrypoints; browse/drop/retry/reanalysis; one-layer auto-start; explicit multi-layer selection; cancel decision; multiple outer files; busy state; native path quoting. |
| `test_gis_exports.py` | Names/IDs/provenance and omission counts survive JSON/XLSX/KMZ and state splits; point grouping by layer; legacy KML exports remain compatible. |
| `test_gis_customer_archives.py` | Hash-pinned full supplied archives, exact structural baselines, independent per-feature mileage, actual controller routes. |

Generate tiny synthetic fixtures with known coordinates and independently stated
expected topology/counts. Include a real multi-layer GPKG with line, point, polygon,
and nonspatial contents, plus a mixed GeometryCollection layer. Test declared
generic geometry containing unsupported curves. GPKG coverage cannot be inferred
from the supplied shapefiles. If a writer is used to build fixtures, keep its
version/generator and separately verify the expected records; do not derive
expected results using the production normalization code under test.

For cross-format equivalents use identical source vertices where possible. For
projected transformations assert independently calculated locations/lengths at a
declared numeric tolerance and report datum accuracy separately. Do not allow a
meter-scale tolerance to hide connected multipart gaps or dropped vertices.

Preserve existing parser, point, folder, repair, state allocation, overlap,
workload, cancellation, export, and native dialog suites. Expand the existing
[route tests](tests/test_state_input_routes.py),
[session tests](tests/test_repair_integration.py), and
[packaged smoke](src/pipeline_calculator/smoke.py) instead of inventing parallel
test harnesses. Run focused checks per change, then the repository's full
non-native suite and required Windows/macOS native suites at integration.

### Distribution release gates

Update the actual [Windows build script](scripts/windows/build_exe.ps1) and
[macOS build script](scripts/macos/build_app.sh), not only the historical `.spec`.
Bundle the approved reader, native libraries, GDAL/PROJ data, and required hooks.
Ensure pyproj and GDAL do not accidentally share incompatible data directories.
Record shipped versions, supported drivers, app size/startup impact, and measured
memory/cancellation capacity.

Require offline SHP/ZIP, GPKG layer selection, GeoJSON, and projected-coordinate
smoke tests in the frozen Windows EXE and Apple Silicon app. Retain current
macOS 15/26/27 verification and post-signing checks. A development import alone
does not establish distribution support. Release only after archived/signed
artifacts pass the same import assertions and ordinary KML regressions.

## 8. Delivery sequence and completion criteria

| Work package | Deliverable and exit gate |
| --- | --- |
| 0. Evidence and backend feasibility | This sample inventory; then a pinned reader spike proving raw geometry fidelity, projected CRS, bounded memory/cancellation, and frozen Windows/Apple Silicon imports. Resolve reader/cap limits before promising support. |
| 1. Shared import contracts | Registry, source/session/provenance interfaces, KML adapter, desktop + headless routing; existing KML behavior passes unchanged. |
| 2. Vector readers and normalization | SHP/GPKG/GeoJSON, geometry/CRS policies, typed diagnostics; synthetic cross-format tests and original ZIP geometry baselines pass. |
| 3. Packages and import decisions | Bounded ZIP handling, sidecar capture, layer/CRS selection, direct drop/browse/retry, lifecycle; supplied ZIPs need no manual extraction or layer dialog. |
| 4. Results and regression coverage | Generic provenance/coverage in UI and all exports; full customer/controller/state and existing regression suites pass. |
| 5. Packaged release | Offline Windows and signed Apple Silicon validation, user documentation, tested capability list and capacity limits. |

Work packages 2 and 3 can proceed in parallel after contracts are fixed, sharing
one fixture corpus. Do not enable an extension in the released UI until its whole
route and packaged driver are verified. An early SHP/ZIP milestone is possible,
but it must retain these shared contracts so GPKG does not require a second import
architecture.

Definition of done: a user can drag either original ZIP into a packaged app,
obtain the verified supported geometry with honest omission/projection metadata,
reanalyze without reopening the original, and export matching results. A
multi-layer GPKG supports deliberate layer selection, and all advertised formats
follow the same geometry and measurement rules without regressing KML/KMZ.

The original planning change prepared this specification, independent
sample-inspection script/report, and byte-identical regression fixture copies.
The implementation now adds the shared import package, desktop decisions,
provenance/export integration, regression tests, pinned dependencies, and build
hooks. See the [implementation report](docs/validation/gis-import-implementation.md)
for executed checks and platform-specific release evidence.
