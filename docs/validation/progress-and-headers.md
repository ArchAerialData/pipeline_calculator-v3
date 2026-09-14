# Measured progress and table headers

Verified locally on Windows on 2026-09-14.

## Behavior

- The runtime advisory now uses observed processing speed, with a threshold of
  **more than 60 seconds**, instead of a segment-count or density threshold.
  It projects elapsed time plus unfinished work in the current stage, using at
  least three seconds of measurements and a sustained high estimate over one
  second with advancing counts. Processing that already exceeds 60 seconds also
  qualifies. The warning may appear partway through a run; unstarted stages do
  not have a measured speed, and the estimate cannot guarantee completion time.
  Accepting once suppresses further advisories in that run. Hard limits remain.
- The green determinate bar advances from actual completed work within fixed
  stage shares. It does not animate on a timer or represent percent elapsed time.
  It holds during indivisible operations and reaches 100% when processing finishes.
  The yellow Cancel button sits below it.
- Elapsed time starts in the worker, excludes warning-decision waits, freezes at
  completion, and resets with the bar for a new import.
- Sortable headers have persistent filled up/down indicators at the right edge,
  neutral paired arrows before sorting, and a single arrow for the selected
  direction. Header borders, icon size and spacing scale with DPI. Existing raw
  numeric sorting, pinned totals, pagination and corridor identity are preserved.

Implementation: [progress model](../../src/pipeline_calculator/core/progress.py),
[execution context](../../src/pipeline_calculator/core/execution.py),
[processing screen](../../src/pipeline_calculator/gui/controllers/analysis_session.py),
[shared sort headers](../../src/pipeline_calculator/gui/sorting.py), and
[table styling](../../src/pipeline_calculator/gui/tables.py).

## Evidence

- Full suite: **280 passed in 121.94 seconds**, including native GUI layout,
  cancellation, warnings, numeric sorting and corridor-launch regressions.
  Log: `.validation-output/progress-headers-full-tests.log`.
- A real asynchronous WWM GUI run completed in **11.594 seconds**, with **zero
  runtime warnings** and 102 observed progress snapshots. Progress was monotonic
  and reached 100%. A second input in the same app reset the timer and bar and
  completed successfully. No Tk callback exceptions occurred.
  Evidence: `.validation-output/progress-headers/real-wwm-progress.json`.
- Inspected native captures of the live progress screen, neutral and sorted
  pipeline headers, and the settled WWM overlap page. Corridor buttons remain in
  their Action cells. A separate native probe checked all 14 visible button
  rectangles; offscreen rows remained hidden. Captures and geometry evidence:
  `.validation-output/progress-headers/`.
- Native tests and packaged checks run on a private, undisplayed Windows desktop.
  They do not switch the user's desktop or request foreground activation.
- Packaged startup, WWM smoke results, replacement paths and SHA-256 hashes are
  recorded in `.validation-output/progress-headers-replacement.json` after build
  verification and local replacement.

## OBJECTID

The [parser](../../src/pipeline_calculator/parsers/kml_kmz.py) reads the exact
`OBJECTID` attribute from KML `<Data name="OBJECTID"><value>…</value></Data>` or
`<SimpleData name="OBJECTID">…</SimpleData>`. If neither contains a value, it
displays `N/A`.

Direct inspection of `Q3 - WWM Pipelines.kmz` found **zero Data, SimpleData and
SchemaData elements**. Its first placemark has a separate `id="ID_280000"`, and
its HTML description contains `RouteId = BRD`. Those are different identifiers;
the parser does not label them as OBJECTID. No parser change was needed for this
request. Other KMZs must be inspected individually to determine what they export.

Input SHA-256:
`b927593def8f50ff08028f00fd141ca6f0d7483c2331f9df9731d16735a3f937`.
Field inspection is saved in
`.validation-output/progress-headers/objectid-evidence.json`.
