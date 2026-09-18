# Verified KML/KMZ repair implementation

September 17, 2026. Implementation and local Windows validation complete.
Product and correctness requirements are in
[the repair plan](../../KML_REPAIR_FEATURE_PLAN.md).

## Delivered workflow

Browse/drop prepares an immutable in-memory source snapshot. Eligible input
finishes preparation with a **Repair & analyze** decision, without calculating
results first. Approval starts a new analysis job using the captured source and
settings. Explicit Browse/drop acquires a new snapshot; Retry/Adjust Parameters
reuse verified bytes, with fresh normalized feature records for each run.

The four rules are missing schema-instance metadata namespace, XML whitespace
before the declaration, unambiguous literal ampersands in restricted display
metadata, and a valid KML/KMZ carrying the wrong top-level extension. Other defects
receive specific source, support, or operational feedback. Client requests are
copyable and do not recommend deleting shapes or guessing vertices.

The notice appears once above the result selector/tabs and remains available after
an analysis failure/cancellation. Optional Save repaired copy stages the retained
snapshot, verifies ordinary reimport, and atomically publishes without replacement.
Unsupported relocation, changed primary selection, and source aliases prevent
saving. JSON and ordinary/state workbooks carry the same repair provenance.

## Preservation and bounded processing

- Exact, reversible byte patches; independent edit-context and structure checks.
- Geometry subtrees unchanged, including unmeasured points/polygons and altitude.
- Independent feature/path inventory compared with actual application extraction;
  wrong case/namespace, ambiguous membership, and invalid tuples block repaired
  analysis even when the ordinary parser would overlook a portion.
- Frozen primary member, link traversal and source identities; retained originals
  and all archive resource hashes. Repaired sizes cannot silently select a child
  document in place of the original entry point.
- XML Base, XInclude, signed XML, update delta documents, and ambiguous NetworkLink structures block repaired
  analysis. A graph with two different plausible linked targets is tested;
  merely disabling Save would not prevent wrong mileage during the session.
- No DTD/entity expansion, remote retrieval, permissive XML recovery, geometry
  cleanup, or temporary source-file round trips for calculation.
- Bounded reads, archive decompression, XML complexity, diagnostics, patch count
  and patch size; cooperative cancellation. Existing valid ordinary-input
  geometry interpretation remains unchanged. Earlier DTD/complexity rejection
  and archive preflight are deliberate input-policy safeguards.

Implementation entry points:

- [Document rules and verifier](../../src/pipeline_calculator/parsers/repair.py).
- [Source acquisition, coverage, immutable baseline and save](../../src/pipeline_calculator/parsers/source.py).
- [Analysis orchestration](../../src/pipeline_calculator/gui/controllers/analysis_controller.py).
- [Shared desktop recovery and notice](../../src/pipeline_calculator/gui/repair_ui.py).
- [Workbook/JSON provenance validation](../../src/pipeline_calculator/export/repair_provenance.py).

Headless callers can use `prepare_source(path)`, inspect its report, explicitly
call `approve()`, and pass `fresh_parse()` to `PipelineAnalyzer.analyze_parsed()`.
The controller's `analyze_file(..., approve_repair=True)` is an explicit convenience
opt-in. Ordinary direct parsing/analysis remains strict by default.

## Verification evidence

| Check | Result |
| --- | --- |
| Broad non-native suite, excluding fixture-authoring tools | 444 passed, 40 native cases deselected; 154.83 seconds |
| Focused engine/source/integration/state-progress/workbook/state-analysis suite after additional audit fixes | 195 passed; 3.71 seconds |
| Final engine/source/integration/packaged-smoke suite after linked-document guards | 206 passed; 2.70 seconds |
| Production preparation of supplied August 2023 KMZ | 42 pipelines, 44 paths, 2,139 vertices; 920,747.8695152604 meters / 572.1250571784351 US survey miles; original hash unchanged |
| Native repair and surrounding UI regressions | 26 passed; latest 7 repair cases rerun after visual refinements. Both GUI entrypoints, keyboard/focus, small windows and high DPI exercised |
| Offline frozen Windows application, both entrypoints | Passed; repair approval gate, source preservation, geometry verification, saved-copy reimport, provenance export, 51 boundary jurisdictions and state-map roundtrips; no Tk callback errors |
| macOS packaged execution | Not run on this Windows host; remains a release gate for macOS distribution |

The verified Windows executable was published to both
`dist/Pipeline_Calculator_v4.exe` and
`dist/Pipeline_Calculator_v4.22-dev.97989716af38.dirty.exe`.
Both copies match the tested artifact: 102,572,973 bytes, SHA-256
`9745fcdcdd406a58b00d9fee8cb8958fdc94c4654ad4cb0a7190973a73ec6f3b`.
The version carries `.dirty` because these changes are not committed.
See the retained [Windows package verification](kml-repair-windows-package.json).
Native source-level UI tests exercise the repair dialogs; the frozen smoke checks
exercise both application entrypoints and the packaged repair backend, saving and
export paths. No customer source file was overwritten or added to the repository.

The supplied-file result is recorded in
[production sample evidence](kml-repair-production-sample.json). It validates
production preparation, approval, geometry coverage and original mileage; that
specific check does not run overlaps or state analysis. Synthetic end-to-end
fixtures compare original mileage, overlap output and Texas/Oklahoma results to
independently authored valid counterparts with identical settings. See
[integration tests](../../tests/test_repair_integration.py),
[engine tests](../../tests/test_kml_repair_engine.py), and
[source tests](../../tests/test_kml_repair_sources.py).

A subsequent [stored-mileage audit](august-2023-mileage-comparison.md) confirmed
that all supplied geometry is included, but four source records account for
99.963% of the difference between 648.601713 stored miles and 572.125057 calculated
miles. Matching the unverified attribute total is therefore not a repair
acceptance criterion; complete client-system coverage needs clarification from
the source provider.

Tests also cover no-op valid inputs, combined defects, near-miss refusals,
adversarial edit ledgers, duplicate/unsafe archive members, corrupt resources,
reimport selection changes, snapshot changes during acquisition, invalid/short
geometry, cancellation, source mutation after approval, output collisions, and
JSON provenance that cannot be silently converted to strings.

Repairing formatting does not reconstruct missing source data or establish that
the client exported their entire system. Invalid or ambiguous cases remain
blocked with observed issues and honest inspection limits.
