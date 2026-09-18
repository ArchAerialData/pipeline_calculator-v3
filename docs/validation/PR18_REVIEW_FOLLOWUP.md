# PR 18 review follow-up

Reviewed September 18, 2026 against merged main `29b997bbd6b8befbd44c944a882dc5a1dca80084`.
Source: [Copilot review](https://github.com/ArchAerialData/pipeline_calculator-v3/pull/18#pullrequestreview-5251812974).
The review contains a summary but no inline review threads to resolve.

## Confirmed and corrected

- **Optional linked-KML diagnostics:** ordinary XML syntax errors already produced
  `linked_kml_parse_error`. The narrower failure was an unsupported DTD in an
  optional linked document: `safe_xml_root` raised `RepairFailure`, aborting the
  direct parser. Both loose KML and KMZ cases failed before the fix. The parser
  now matches existing source-session handling, preserving healthy sibling paths
  while reporting an error and explicitly incomplete analysis. Primary-document,
  resource-limit, policy and ambiguous-geometry failures remain fatal. Repair
  still refuses to certify missing linked geometry. See
  [regressions](../../tests/test_linked_kml_diagnostics.py).
- **Corridor launch availability:** a canonical section marked `ready` but missing
  polygon data incorrectly enabled its map action. The serializer already rejected
  the payload before writing or launching a map, so this was a UI availability
  mismatch, not an empty-map export or mileage error. Canonical sections now need
  a nonempty polygon list; state-clipped geometry takes precedence. Valid multipart
  polygons, holes and legacy geometry remain supported. Nine new cases failed
  before the correction; launch-route, native-button and serialization checks pass.
- **Feasibility status:** the old README incorrectly said production implementation
  had not started. It now identifies its measurements and baseline as historical
  and links to the later implementation/release evidence.

## Findings not accepted as written

- **Lazy-table sorting:** both Combined and State tables already have empty heading
  commands during loading. `HeaderSorter` is installed after every source row and
  TOTAL have been inserted. An isolated native probe used 6,000 rows per scope:
  the first 200 rows had no active header commands; an early header click preserved
  source order without a callback error. Completed 6,001-row tables sorted numbers
  correctly in both directions with TOTAL pinned. Early clicks are intentionally
  ignored, not applied to an incomplete subset. No production change was made.
  Local evidence: `.validation-output/pr18-review/lazy-pipeline-sort.py` and `.json`.
- **Repair export guidance:** `repair Details` refers to the existing **Details**
  action in the app's repair notice (`RepairWorkflow.show_details`), not a workbook
  sheet. The **Analysis Details** sheet is already where the summary cell appears;
  directing users back to it for full nested evidence would be misleading. A probe
  verified the summary sheet, the report passed to the app's details formatter,
  and exact repair-report preservation in JSON. The GUI caps its text preview at
  24,000 characters; JSON retains the full report. The suggested sheet-name
  replacement was not applied.

## Verification

The combined focused run passed **302 tests in 26.97 seconds**, covering linked
documents, structural guards, repair/source integration, corridor launch and
visibility, export consumers, table sorting, and workbook contracts. Native GUI
cases ran through the existing isolated-process harness. Command:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_linked_kml_diagnostics.py tests/test_kml_structure_safety.py tests/test_kml_repair_sources.py tests/test_kml_repair_engine.py tests/test_repair_integration.py tests/test_corridor_launch.py tests/test_corridor_visibility.py tests/test_corridor_consumers.py tests/test_table_sorting.py tests/test_followup_audit.py tests/test_export_workbook.py -q
```

Output: `.validation-output/pr18-review/focused-tests.txt`. No mileage, overlap
qualification, state allocation, corridor construction, or geometry-repair rules
were changed by this follow-up.
