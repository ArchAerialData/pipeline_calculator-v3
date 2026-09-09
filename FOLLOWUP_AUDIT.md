# Follow-up implementation audit

Audited the clean working tree at `39f3260` (Bug Fixes). The changes in this
follow-up remain uncommitted for review. No branches were merged or deleted,
and no release was published.

## Numerical findings and decisions

1. **Inconsistent groups of three pipelines.** Three 300 m parallel lines at
   offsets 0, 10, and 20 m with a 15 m limit previously removed 500 m, leaving
   an implausible 400 m effective total. The adopted policy requires every pair
   within a group to qualify, rather than joining a chain by transitivity. The
   result now removes 300 m and retains two 300 m survey passes. Six input-order
   permutations are tested. Four mutually compatible lines and two independent
   pairs are tested separately.
2. **Offset midpoint samples hid real overlap.** Lines 2 m apart, sampled at
   different longitudinal positions, previously disappeared with a 3 m range.
   Candidate search now allows the finite segment extent and tests longitudinal
   overlap and transverse separation using their local tangents. Half-segment
   and smaller offsets are covered, as is an end-to-end non-overlap control.
3. **Dateline corridor center near Greenwich.** A 300 m corridor crossing 180
   degrees previously had a longitude near zero. Bounds now unwrap around a
   local origin; centerlines and polygon coordinates use geodesic forward/inverse
   operations. Tests bound all polygon points and the center near the source
   for east/west dateline crossings and a near-pole example. Wrapped bounding
   boxes may intentionally have western longitude greater than eastern longitude.
4. **Suppressed geometry failures.** Segmentation could substitute a made-up
   midpoint/bearing after a geodesic operation failed. Those failures now stop
   the calculation and surface through the existing incomplete/error handling.
   Non-finite analysis parameters and invalid minimum lengths are rejected.

Implementation: [bundling](src/pipeline_calculator/core/bundling.py),
[overlap](src/pipeline_calculator/core/overlap.py),
[segmentation](src/pipeline_calculator/core/segmentation.py).

The grouping implementation forms disjoint groups of sampled segments, allows
only one segment per pipeline in each group, and requires every cross-pair to
appear in qualified overlaps. It merges nearest candidates first with geometry
tie breaking, independent of the order of pipelines in the file. It is a
deterministic conservative heuristic, not a globally optimal clique cover or
flight-route solver. Original pipeline geodesic lengths are unchanged.

## Input and export findings

- Missing/malformed linked KML and empty supported-feature sets previously could
  return `analysis_complete=True`. Missing input now makes the status incomplete
  while retaining known valid geometry and its diagnostics. Intentionally ignored
  polygon geometry and unreferenced archive documents retain their separate
  diagnostic behavior; the parser does not invent pipeline mileage from them.
- KML reads are bounded before XML parsing: 64 MiB per decompressed document,
  256 MiB total, 1,024 documents, 10,000 archive entries. Traversal uses a deduplicated
  queue. Duplicate normalized KML names are rejected rather than silently choosing
  one archive member. Existing archive-root containment remains in place.
- Geometry processing stops explicitly at 1,000,000 generated segments or
  5,000,000 neighbor candidate inspections. Tiny segment sizes and extremely wide
  search radii cannot expand the work indefinitely without a clear failure.
- An input name `=1+1` was exported as an Excel formula. Source fields now remain
  literal text. The application's deliberate mileage-total formula is preserved.
  A save/reload test checks the actual XLSX cell types, including diagnostics.

Implementation: [parser](src/pipeline_calculator/parsers/kml_kmz.py),
[Excel exporter](src/pipeline_calculator/export/xlsx.py).

## GUI fixes

- Busy guards prevent reentrant imports/reanalysis from launching competing jobs
  or overwriting the selected file. Old results are cleared at the start, and
  failed startup resets the busy state. Both entrypoints are covered by the guard.
- Clamped input fields now show the actual values used, including legacy GUI
  inputs. Summary parameters are shown once; degree labels use readable text.
- Overlap rows beyond 20 are now reachable through Previous/Next controls. Pages
  retain global section indices, show their range/count, and bound widget creation
  to 20 rows. Horizontal scrolling supports narrow layouts. Offscreen corridor
  buttons are hidden rather than drawn over unrelated cells.
- The overlap footer explicitly says its total is pairwise, rather than implying
  that summing it gives project savings.
- Incomplete-analysis notices show a bounded list of distinct messages and point
  to diagnostics for additional issues.
- The legacy GUI now delegates summary and overlap rendering to the same modules
  as the default GUI, removing duplicated implementations and keeping fixes aligned.

Implementation: [main window](src/pipeline_calculator/gui/main_window.py),
[overlap tab](src/pipeline_calculator/gui/tabs/overlap_tab.py),
[summary](src/pipeline_calculator/gui/tabs/summary_tab.py),
[parameter state](src/pipeline_calculator/gui/state.py).

## Verification

- `python -m pytest -q`: **106 passed** on Windows (79 existing plus 27 new cases).
- Native Windows widget smoke: 45 sections paginate as 20 / 20 / 5; Next disables
  on the last page and Previous restores the correct rows. This test is skipped
  on non-Windows CI; the numerical and headless state tests run everywhere.
- Tests cover the numerical reproductions, input order, offsets, true non-overlap,
  dateline/polar coordinates, incomplete input, compressed-input limits, document
  budgets, segment/candidate limits, duplicate archive names, geometry errors,
  GUI busy-state cleanup, actual pagination widgets, and persisted XLSX cell types.
- `python -m compileall -q src` and `git diff --check`: passed.

Tests: [follow-up cases](tests/test_followup_audit.py),
[original regression cases](tests/test_calculation_regressions.py).

## Remaining limits and worthwhile next work

- Numerical results remain **sampled estimates**. Segment length, short trailing
  remainders, and sharp bends can affect boundary mileage. Finite tangent comparisons
  do not constitute an exact geometric union of every original polyline edge.
- Mutually compatible grouping enforces the chosen separation rule; it does not
  model aircraft repositioning, terrain, camera coverage, or flight-path continuity.
- Real project KMZ comparisons and long, curved/branching corridor examples are
  still needed before calling the application survey-validated. No representative
  production KMZs were available in the repository during this audit.
- Native widgets were exercised, but full visual inspection of packaged Windows
  and macOS apps and exported Google Earth corridors remains outstanding. The
  dateline tests verify numerical coordinates, not Google Earth's rendered topology.
- Useful next improvements are cooperative cancellation with measurable progress,
  profiling large real datasets, and a reference calculation using exact covered
  intervals to quantify sampling error. These require separate implementation work.
- Input budgets are guardrails, not a claim of comprehensive security certification
  for all XML/ZIP inputs or of maximum resident-memory bounds.
