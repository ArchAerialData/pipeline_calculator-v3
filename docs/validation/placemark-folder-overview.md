# Placemark folder overview and export provenance

October 2, 2026. Source changes extend the Antero archive compatibility fix.

## Behavior

The Combined Placemarks page replaces individual rows with a combined point
total and cards for each immediate containing folder. Each valid KML Point
geometry counts once, including multiple points within a single Placemark.
Line vertices and folders containing only polylines do not create pin cards.
The overview includes a measured zero when the point inventory is empty.

Nested cards show their ancestor path. Root points use **No subfolder**; older
results without provenance use **Folder not recorded**. Blank folder names use
**Unnamed folder**. Source-local folder IDs keep equal-named siblings separate;
source KML identities distinguish folders in different linked documents.
NetworkLinks load each reachable document once using its own physical folders.
Document titles are not treated as Folder names. Existing hidden-feature
inclusion behavior is unchanged. Groups follow source encounter order.

XLSX tables append **Subfolder**, **Folder Path**, and **Source KML** to combined
polylines, Point Pins, State Pipeline Lengths, and Shared Borders. Existing mileage
column positions/formulas remain unchanged. Source strings remain literal text,
including names starting with `=`. Folder metadata follows retained snapshots,
approved repairs, state fragments, save/reimport, JSON results, and XLSX packages.

The folder cards reuse the Summary page's card palette, typography, deferred
layout, wrapped labels, and the shared scroll frame. Card columns reflow with
the available logical width; labels wrap without fixed card heights. Large
folder inventories load in batches and pause when the page is hidden.
The viewport is keyboard-focusable with arrow, Page Up/Down, Home, and End
navigation; its scroll bindings remain scoped to that page.

## Direct source evidence

The retained [original Antero fixture](../../tests/fixtures/geography/antero_midstream_data.kmz)
has 7,275 valid points. Independently reading its XML yields these counts:

| Immediate folder | Points |
| --- | ---: |
| Antero Waterline Facility | 4,192 |
| Antero Gas Pipeline Facility | 1,906 |
| Interchange Locations | 420 |
| Launcher Receiver | 477 |
| Antero Pads | 250 |
| Compressor Stations | 30 |
| Combined | 7,275 |

The two pipeline folders contain 639 gas features and 321 water features.
The [real-file regression](../../tests/test_antero_kmz.py) checks these exact
counts as well as every original line path/vertex and unchanged file hashes.
Three source records have no geometry and retain their existing warnings; they
do not contribute invented pins. See the [original archive audit](antero-kmz-audit.md).

## Validation scope

- Final broad core run: **962 passed, 14 skipped, 94 native cases deselected**
  in 123.70 seconds. The exclusions were native UI (run separately), fixture
  authoring tools, and Windows layout probes (run separately). Skips were five
  optional fixture-author audit modules and nine macOS bundle checks lacking
  `macholib` on this host; they are not macOS compatibility passes.
- [Folder provenance tests](../../tests/test_folder_provenance.py): nested,
  unnamed, duplicate-name, linked, mixed, multipoint, repaired and state inputs;
  corruption of folder/source attribution blocks independent repair verification.
- [Folder summary tests](../../tests/test_placemark_folders.py): disjoint totals,
  source identities, zero versus unavailable, and detached metadata copies.
- [Workbook tests](../../tests/test_folder_exports.py): save/reload, literal
  names, root and legacy records, combined/state workbooks, shared fragments,
  and exported packages.
- [Native folder UI tests](../../tests/test_placemark_cards_ui.py) render the
  original Antero fixture at 100%, 125%, 150%, and 200% scaling across physical
  1366x768, 1920x1080, 2560x1080, and 640x640 windows. All sixteen configurations
  passed width/height, wrapping, scrollbar, totals, and last-card reachability
  assertions. [Measured layout receipts](placemark-folder-layout.json) retain
  requested and actual sizes; screenshot capture can be enabled with
  `PIPELINE_PLACEMARK_EVIDENCE_DIR`.
- Native tests also cover long/duplicate names, nested/root/legacy groups,
  zero/unrecorded inventories, keyboard scrolling, 120-folder loading with
  hide/remap, DPI changes, and destruction with queued callbacks. Surrounding
  tests exercise Summary typography, state scope replacement, shared scrolling,
  and modern/legacy entrypoints including narrow windows up to 250% scaling.
- Final UI validation: six folder-card cases and the 100-scope replacement
  lifecycle case passed together (**7 passed, 120.86 seconds**). The other
  **60 surrounding UI/layout cases passed** in the broader run. Its original
  lifecycle assertion expected only Summary's scroll handler; it now checks
  the two live scroll frames and proves retired Summary and Placemarks views
  are released. There are no remaining failures in these selected suites.
- A final 1500x800 Windows preview was rendered from the original Antero
  fixture and visually inspected: combined total and all six cards fit without
  clipped text. Local screenshot evidence is retained under
  `.validation-output/placemark-cards/preview.png`.
- The Windows and macOS build scripts both run the full pytest suite, including
  portable `native_gui` tests. Native macOS execution requires the macOS CI
  runner; it cannot be validated on this Windows host. Windows-only probes
  supplement rather than stand in for that native macOS gate.
- No executable or macOS app bundle was rebuilt for this source-level change.

The subsequent macOS CI run exposed a shared viewport teardown leak during
repeated view replacement. See the [follow-up investigation and fix](placemark-scroll-lifecycle.md)
for reproduction evidence and the updated validation scope.
