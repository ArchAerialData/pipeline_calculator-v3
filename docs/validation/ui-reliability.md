# Summary and shared UI reliability

Implementation baseline: `a8640fd` (State Aware Planning). This change addresses the [UI resolution plan](../../MACOS_SUMMARY_UI_RESOLUTION_PLAN.md) without changing geography calculations or export formats. Final machine-readable outcomes are retained in [ui-reliability-evidence.json](ui-reliability-evidence.json) alongside this report.

## Final Windows results

- Full suite: **366 passed**, no failures, errors or skips (201.96 seconds).
- Strict reference benchmark: 100 Summary returns, including 95 warm returns at **42.61 ms p95 / 45.38 ms maximum**; resize reflow maximum 281.87 ms. No callback errors.
- One hundred scope replacements: no retained retired views, no callback errors, and maximum heartbeat gap 159.79 ms. A complete replacement plus layout took up to 518.67 ms; this is separate from warm tab return.
- Large-table and background-export heartbeat gaps: 14.23 ms and 20.08 ms respectively.
- Final Windows executable: both modern and legacy packaged smoke checks passed, including 20 Summary return/disclosure cycles each, offline geography validation, and empty stderr. Artifact version `4.17-dev.a8640fd783e9.dirty`; its SHA-256 is recorded in the evidence file.

The strict reference measurements preceded the final hidden-page focus-transfer addition; that addition received a targeted native check, and the final package includes it. Changes are uncommitted. The other task's state-boundary audit documents and fixtures were preserved.

## Changes and evidence

| Area | Implementation | Verification |
| --- | --- | --- |
| Additional details | [DisclosureButton](../../src/pipeline_calculator/gui/disclosure.py) uses Tk-painted text/background instead of the OS-drawn button face. Supports pointer cancellation, Return/Space, focus, wrapping, and disabled colors. Details are constructed once, on first expansion. | Native input tests, Summary typography tests, both packaged GUI modes, and Windows screenshots. Actual macOS pixels remain a release gate. |
| Resize/navigation stall | [ResultPages](../../src/pipeline_calculator/gui/layout.py) defers tab/menu layout outside Configure callbacks; selecting the current tab does not unmap it. Summary layout is coalesced and cancelled while hidden, but cards/labels receive a valid initial grid immediately. | A new minimal navigation test originally timed out after 45 seconds inside CTkOptionMenu redraw/update_idletasks. A high-DPI scope test also exposed unplaced one-pixel cards while waiting for a timer; immediate initial layout removes that dependency. Repeated Summary returns verify content bounds, retained widgets, and settled callbacks. These are not proof of the exact original macOS trigger. |
| Retained callbacks | [Owned bindings](../../src/pipeline_calculator/gui/bindings.py) detach only each widget's own callbacks. Wrapped labels release parent subscriptions; scroll frames release global wheel callbacks. Shift state comes from each wheel event. | Twenty create/destroy cycles release all weak references and restore binding scripts, including an unrelated live listener. One hundred scope replacements retain no retired Summary views. |
| Shared scrolling | [AutoScrollFrame](../../src/pipeline_calculator/gui/scrolling.py) owns deferred scrollbar updates, remap refresh, cancellation, and nested scrolling. Settings, parameter/progress bodies, corridor dialogs and package export use it. | Small/high-DPI layouts, warning/progress transitions, dialog tests, nested-table wheel tests. A nested table consumes wheel input even at its edge; scroll the outer page by pointing outside the table. Unknown native popup widget paths are ignored safely. |
| Scope replacement | [Results page](../../src/pipeline_calculator/gui/pages/results_page.py) prepares the replacement before removing the valid view. Same-scope selection is a no-op. Construction failure preserves the previous scope and exposes Retry display. | Injected render failure, selector/view agreement, 100 replacements, and DPI changes. No pack reference to a destroyed view is retained for CTk's DPI replay. Scope reconstruction remains more expensive than ordinary tab return, but the old results remain available while it is prepared. |
| Native styles | [Shared styles](../../src/pipeline_calculator/gui/styles.py) select clam without repeatedly resetting it and reuse styles by pixel dimensions. Scope popup colors/fonts are applied to its own listbox. | Selected/readonly/disabled state lookups, two-window size isolation, and style counts after 100 scope replacements. Actual Aqua appearance remains to be checked. |
| Large tables | [TableLoader](../../src/pipeline_calculator/gui/table_loading.py) populates at most 200 records or approximately 8 ms per batch, pauses when hidden, and closes the iterator/cancels callbacks on destruction. Loading failures show a message. Pipeline sorting uses one children-reorder operation. | 20,000-row completion, heartbeat responsiveness, pause/resume, cancellation and injected load failure; existing full-precision sorting and total-row tests. Small tables still populate immediately. |
| Export responsiveness | [Ordinary export](../../src/pipeline_calculator/gui/actions/export_actions.py) uses a [background file action](../../src/pipeline_calculator/gui/background_action.py) with a responsive progress window. Writes preserve the existing XLSX/JSON functions. Parameter transition errors are shown rather than silently discarded. | Worker-thread identity, heartbeat and write-failure tests, existing workbook/JSON/package tests. Profiling before the change measured about 2.49 seconds of synchronous XLSX writing for a synthetic 50,000-row result. |
| Packaged regressions | [Smoke mode](../../src/pipeline_calculator/smoke.py) runs 20 Summary return/disclosure cycles in both actual GUI implementations, in addition to existing offline analysis/geography checks. The redundant preliminary legacy Tk root was removed. [Packaging validation](../../scripts/validation/check_packaged_smoke.py) rejects Tk callback errors printed on stderr. | Source and frozen-app checks in both GUI modes. Tests reject a passing JSON report accompanied by known Tcl callback-error text. |

The scroll adapters depend on CustomTkinter internals, so [requirements.txt](../../requirements.txt) now pins the tested 5.2.2 release. Any dependency upgrade should rerun the lifecycle, scaling, nested-input and packaging checks rather than silently assuming compatibility.

The per-combobox popup listbox path was checked against both the installed Tcl/Tk source and the upstream [Tk 8.6 combobox implementation](https://raw.githubusercontent.com/tcltk/tk/core-8-6-branch/library/ttk/combobox.tcl). This verifies the adapter's structure, not macOS pixel rendering. Ordinary tab changes also transfer focus away from controls in the page being hidden.

## Running the checks

Run the normal suite from the repository root:

```powershell
& ./.venv/Scripts/python.exe -m pytest -q --junitxml=.validation-output/ui-reliability/full-suite.xml
```

Run the strict performance checks separately on reference hardware, without a concurrent build:

```powershell
$env:PIPELINE_UI_PERFORMANCE_GATE='1'
$env:PIPELINE_UI_REPORT_DIR='.validation-output/ui-reliability/reference-benchmark'
& ./.venv/Scripts/python.exe -m pytest tests/test_ui_lifecycle.py tests/test_ui_shared_surfaces.py -q
Remove-Item Env:PIPELINE_UI_PERFORMANCE_GATE
Remove-Item Env:PIPELINE_UI_REPORT_DIR
```

Strict thresholds remain 100 ms p95 / 250 ms maximum for warm tab return, 500 ms for resize reflow, and 250 ms maximum heartbeat gap. The 100-cycle tab test separates 95 warm returns from five returns after changing window geometry. Normal CI uses a two-second watchdog threshold for these measurements, along with the existing external native-process timeout and correctness assertions. This distinction avoids treating a busy desktop as a reproducible benchmark; it does not change the UX targets.

These automated measurements verify mapped content and event-loop progress, not physical display paint. During development, whole-suite/desktop-load runs showed occasional resize/dispatch spikes above strict targets (including approximately 604 ms reflow and a 317 ms heartbeat gap). Do not use a single passing reference run to promise identical latency under every OS workload.

## Packaging and macOS handoff

The isolated Windows build is under `.validation-output/ui-reliability/package/dist/`; previous repository `dist/` artifacts were preserved. The evidence JSON records its hash, version, source hashes, tests and packaged outcomes. Windows visual review covered expanded Summary and parameters; the layout probe also captured import, other result tabs, warning/progress and corridor-dialog surfaces.

Outstanding before declaring the original DMG issues resolved:

1. Identify the reported DMG/build, macOS version, architecture and source KMZ. These were not established in this session.
2. Build the same source for macOS and run the new native tests and packaged smoke gate. The existing macOS CI test command discovers the new cross-platform tests; Windows-only layout tests still do not count as macOS coverage.
3. On a real macOS GUI session, inspect disclosure normal/hover/press/focus states, scope popups, disabled row actions, and return from dialogs in both system appearances. Exercise trackpad input, Retina/display changes, minimize/restore and repeated tab returns using the reported KMZ. Retain screenshots/recording and strict timing results with the artifact identity.

No macOS build or physical macOS visual verification was performed here. The remaining release gate is explicit; neither the Windows executable nor its screenshots can certify Aqua rendering.
