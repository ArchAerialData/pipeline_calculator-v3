# macOS Summary and shared UI reliability resolution plan

Status: implementation and Windows validation complete against `a8640fd`; macOS release verification remains outstanding. See the [implementation and validation report](docs/validation/ui-reliability.md) for dispositions and evidence. The sections below preserve the investigation and accepted plan.
Date: 2026-09-14. Inspected repository HEAD: `212d0d7` plus the active, uncommitted state-breakdown feature.

Expanded audit: 2026-09-14. Includes related controls, scrolling surfaces, navigation, callback ownership, table rendering, and dialog return paths. Verified implementation patterns and measured cleanup gaps are distinguished below from unconfirmed failure hypotheses. The checkout advanced to `a8640fd` (State Aware Planning) during this audit; recheck the current diff and task ownership before implementing.

## Scope and handoff

Resolve two reported DMG defects: the Additional details disclosure becomes a white bar after interaction, and Summary can remain blank or take too long to display after visiting other tabs.

Treat these as two shared reliability workstreams: consistent visible control states, and responsive content throughout page/dialog lifecycles. Include adjacent defects that share those mechanisms. Geometry algorithms, mileage rules, export formats, and general application redesign remain outside this plan; existing feature behavior must be preserved.

The earlier concurrency hold is cleared. The user committed the feature as `a8640fd` (State Aware Planning); the readiness check found only this plan modified, and **Plan state-aware mileage toggle** was idle. The user intends to run that task's implementation audit after this UI work. This investigation changed only this plan. Implement against the committed feature, preserving state selection and combined/state accounting; do not restore UI files from the pre-feature baseline. Recheck status if another task resumes before implementation finishes.

## Readiness decisions before implementation

There is no known blocker to beginning the code changes and automated verification. No additional product decision is required. The following distinctions govern scope and completion:

- **Required work:** disclosure contrast, the measured callback cleanup gaps, shared scroll/lifecycle correctness, safe tab/scope transitions, and regression coverage for every listed related surface. Preserve feature calculations and export contracts.
- **Evidence-dependent changes:** table virtualization/batching, asynchronous ordinary export, broad style refactors, and extra layout scheduling changes require measurements or a demonstrated correctness defect. Audit and test every inventory row; do not automatically rewrite every listed component. A measured pass with no change is an acceptable closure for a hypothesis.
- **macOS completion gate:** this Windows session cannot verify Aqua drawing, actual Retina/display transitions, or the installed DMG. Implement known fixes now; use an actual macOS GUI session and the rebuilt DMG before declaring the reported problems resolved. Missing original DMG identity, macOS version, or the reported KMZ does not block the known fixes, but must remain an explicit reproduction limitation until obtained. Passing Windows tests is not substitute evidence.
- **Dependency compatibility:** [requirements.txt](requirements.txt) currently allows `customtkinter>=5.0.0`, whereas the probes used 5.2.2 and the adapters access private internals. Choose and record the tested compatibility range or a reproducible GUI dependency constraint as part of implementation, test both build platforms, and fail visibly for unsupported adapter assumptions. Do not assume all allowed versions behave alike or introduce an unverified upgrade as the fix.
- **Shared style ownership:** ttk styles belong to the Tcl interpreter. If consolidating per-widget styles, retain a bounded style variant per role and active scale where simultaneous windows need different font/row sizes; a single mutable global size must not resize another window incorrectly. Test multiple live windows at different scales and repeated scope replacement.
- **Performance scope:** retain warm-return targets as proposed acceptance criteria, collect a stable baseline first, and record separate budgets/results for initial rendering and bulk operations. The broader audit is a focused UI reliability pass, not merely a one-button patch. Do not claim timing compliance from callback duration alone.

At implementation start, record `git status`, the exact source/dependency versions, and targeted Summary/state/navigation test results. Preserve any new unrelated changes. Implement and test the required work first, then close evidence-dependent items using measurements. Commit/review this UI change before handing overlapping files back for the other task's feature audit.

The exact downloaded DMG/build and macOS version were not identified. `gh release view` returned `release not found`, and `gh release list --limit 5` returned no entries for the configured origin. The download may be an Actions artifact; this was not established. Do not equate the current checkout with the reported binary.

## Verified findings and remaining uncertainty

### 1. Disclosure contrast: strong platform-specific lead

The disclosure is a native `tk.Button` with near-white foreground `#F1F4F8` and intended dark background `#272D35`. It also specifies white active text. Expanding/collapsing changes the label and geometry; it does not deliberately change the colors. The native button exists in both HEAD and the feature working copy. See [SummaryView and toggle_details](src/pipeline_calculator/gui/tabs/summary_tab.py).

Tk's own tracker documents macOS ignoring button background settings because the OS draws native buttons: [Tk button background issue](https://core.tcl-lang.org/tk/tktview/46274a117823fd65c6d98d38465b43165d0c4680). A light native surface behind the explicit white text is a credible explanation. The exact reported rendering has **not** been reproduced on macOS, so this remains a strongly supported cause rather than a verified screenshot match.

### 2. Summary stalls: layout/remapping needs reproduction

[ResultPages.set](src/pipeline_calculator/gui/layout.py) unpacks every page and repacks the selected existing page. It does not parse KMZs, rerun analysis, or reconstruct Summary. Even selecting the current page performs this remapping. A new background analysis worker or general result cache would not address this observed path.

Summary has several interacting layout mechanisms:

- [SummaryView._arrange](src/pipeline_calculator/gui/tabs/summary_tab.py) changes packing, card grids, and disclosure wrapping synchronously from `<Configure>`.
- `DeferredLayoutFrame` schedules card/text arrangements after 20 ms, with layout-key guards; [WrappedLabel](src/pipeline_calculator/gui/layout.py) independently schedules wrapping.
- [AutoScrollFrame](src/pipeline_calculator/gui/scrolling.py) schedules scroll-position updates after 16 ms and overflow decisions after 25 ms. Adding/removing the scrollbar changes available width, which can change wrapping and content height. No visibility guard or explicit remap refresh is present in this adapter.
- The installed CustomTkinter implementation calls `_draw()` from `CTkScrollbar.set()`, and `_draw()` calls `update_idletasks()`. The adapter already defers scrollbar updates to reduce nested layout; deferral alone does not prove convergence after remapping. Inspected local source: `.venv/Lib/site-packages/customtkinter/windows/widgets/ctk_scrollbar.py` and `ctk_scrollable_frame.py`.

Potential causes to distinguish are repeated layout/scrollbar oscillation, a stale canvas scroll region or offset on remap, callback exceptions, and a platform redraw issue. **No specific cause of the blank Summary is proven yet.**

The incoming feature also rebuilds result tabs when changing Combined/state scope: [results_page.select_scope](src/pipeline_calculator/gui/pages/results_page.py). That is a separate path from ordinary tab switching and needs lifecycle coverage without conflating it with this original report.

### Existing verification

On the live Windows working copy, this command passed **5 tests in 28.75 seconds**:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_ADDOPTS='-p no:cacheprovider'
& ./.venv/Scripts/python.exe -m pytest tests/test_summary_tab.py 'tests/test_ui_layout.py::test_native_layout[1-1800-900-new]' 'tests/test_ui_layout.py::test_native_layout[1.25-640-480-new]' -q
```

These cover estimate states, formatting, typography/scaling, disclosure keyboard activation, and some tab switching. They neither establish macOS pixel contrast nor measure repeated return-to-Summary latency. Native layout tests in [test_ui_layout.py](tests/test_ui_layout.py) are explicitly Windows-only. The working copy was active during inspection, so rerun against a stable feature commit before implementation.

## Implementation sequence

1. **Identify and reproduce the binary.** Record available artifact URL/run ID, commit, app version, macOS version, architecture, display scale, appearance, and bundled Python/Tcl/Tk/CustomTkinter versions. Reproduce with the user's KMZ when available, plus small and large fixtures. Record whether expanding/collapsing or scrolling is necessary. Compare the reported build with the completed feature baseline. Pursue missing macOS evidence alongside implementation; it is not a prerequisite to fixing the already identified defects.

2. **Make the disclosure appearance deterministic.** Replace the native color-dependent button with an application-drawn disclosure using existing CustomTkinter facilities, encapsulated in a reusable control if needed. Explicitly style normal, hover, pressed, focused, and disabled states. Preserve wrapping, visible focus, Tab traversal, Space/Return activation exactly once, focus-driven scrolling, and expanded/collapsed indication. Do not assume CTk supplies the native button's keyboard behavior automatically. Keep the same dark appearance as the surrounding Summary; verify actual pixels on macOS, not just `cget()` colors.

3. **Instrument a bounded tab-cycle reproduction before changing layout.** Capture callback exceptions, selected/mapped page, canvas dimensions, content bounds, scroll fractions, layout/scrollbar callback counts, and click-to-visible timing. Use a mainloop heartbeat to distinguish event starvation from a responsive but blank viewport. Sample stacks on a stall and enforce an external process timeout, following [the existing GUI subprocess harness](scripts/validation/gui_process.py). Avoid diagnostic logging of input geometry or an unbounded log on every configure event.

4. **Fix the demonstrated lifecycle/layout fault.** Keep existing result widgets during ordinary navigation. Make selecting an already visible current page a no-op. If geometry churn is demonstrated, coalesce Summary layout with one owned pending callback, ignore unusable/unmapped geometry, and schedule a fresh layout when shown. Make overflow decisions stable near wrapping thresholds; prove convergence instead of merely increasing timer delays. If scroll state is stale, refresh bounds on remap and clamp the offset while preserving valid user scroll positions. Cancel owned callbacks on destruction. Keep CustomTkinter private API use within the scroll adapter and verify against the packaged version. Do not add `update()`/`update_idletasks()` loops, repeated forced redraws, blanket exception suppression, or background-thread Tk operations.

5. **Optimize construction only if measurements require it.** Detail creation currently scans diagnostics and bundled sections once per Summary construction, including while collapsed. If this affects initial display or state selection for large results, prepare small presentation aggregates once per result/scope and create supporting detail widgets once on demand. Ordinary tab returns must not rescan results, repeat analysis, or recreate widgets. Avoid a global mutable cache or duplicating geometry.

## Regression and release gates

- Add a cross-platform native GUI scenario cycling Summary → each populated tab → Summary at least 100 times. Cover details collapsed/expanded, bottom scroll, narrow and wide windows, scrollbar and card-column thresholds, resize while Summary is hidden, focus changes, and minimize/restore. Check that the visible viewport intersects real Summary content, rather than checking only the outer frame's mapping flag.
- Assert no callback exceptions, no unbounded callback/widget growth, no analysis calls or Summary reconstruction during ordinary tab changes, and correct totals/disclosure state after each cycle.
- Proposed user-experience targets on recorded reference hardware: warm return to visible Summary p95 at most 100 ms, no return over 250 ms, and no mainloop heartbeat gap over 250 ms. Measure the paint/content visibility outcome rather than merely the duration of `set()`. Record initial Summary rendering separately with dataset size. Keep generous external test timeouts so a failure produces evidence instead of hanging CI; do not loosen product targets merely to pass.
- Exercise completed state-breakdown behavior with Combined → state → other tab → Summary → Combined, rapid scope changes, reanalysis, import-new-file, and destruction while layout is pending. Cover both GUI entrypoints.
- Verify disclosure text/focus contrast visually in macOS light and dark system appearance, including hover, press/release, expand/collapse, and focus/unfocus; confirm keyboard usability and Windows appearance. Target text contrast at least 4.5:1 for ordinary text.
- Run the expanded native regression on a real macOS GUI session and the installed packaged app from the resulting DMG, with the reported KMZ. Current [macOS build checks](scripts/ci/macos_build.sh) and Windows-only layout tests are insufficient evidence by themselves. Retain timing data, callback-error report, and before/after screenshots with the artifact's build identity.

Completion requires both defects to pass those checks on macOS. This plan and the passing Windows baseline do not constitute a fix or a verified macOS resolution.

## Expanded audit: related instances to resolve together

### A. Confirmed callback retention, beyond pending timers

Two isolated native Windows probes used Python 3.11.9, Tcl/Tk 8.6.12, and CustomTkinter 5.2.2 through the existing `run_gui` harness, with external 20-second/15-second timeouts. Both probes completed successfully. This audit modified only this plan.

| Probe | Measured result | Interpretation and limit |
| --- | --- | --- |
| Create and destroy 20 `AutoScrollFrame` instances, destroying each containing frame, processing events, dropping Python variables, then collecting garbage | Global binding scripts for MouseWheel, Shift press, and Shift release grew from 0 to 39 lines each; weak references showed 20/20 destroyed scroll frames still retained | Confirms retained bindings/objects in the installed dependency path. Script line counts include blank separators, not 39 separate callbacks. This does not establish the latency impact or reproduce the macOS stall. |
| Create and destroy 20 `WrappedLabel` instances while their CTk parent stays alive | Parent canvas Configure binding grew from 0 to 39 lines; 20/20 destroyed labels remained referenced after garbage collection | Confirms missing detach for this lifetime relationship. Destroying the parent can release these callbacks; do not label every full-page replacement a label leak. |

Reproduction recipe: create one `ctk.CTk` root; record `root.tk.call('bind', 'all', sequence)`; loop over a fresh host and `AutoScrollFrame`, update the event loop, destroy the host, update again, then delete local references and run `gc.collect()`. Retain only `weakref.ref(view)` for inspection. For labels, keep the host alive and inspect `root.tk.call('bind', host._canvas._w, '<Configure>')`, because CTkFrame redirects binding to its canvas. Count retained objects and binding-script growth before/after; do not inspect only Python widget counts or only the native frame's Configure binding.

Sources: [AutoScrollFrame](src/pipeline_calculator/gui/scrolling.py), [WrappedLabel](src/pipeline_calculator/gui/layout.py), installed `customtkinter/windows/widgets/ctk_scrollable_frame.py` (`bind_all` registration and destruction), and `ctk_frame.py` (binding redirection).

Required resolution:

- Give each added timer, parent/root binding, variable trace, and appearance/scaling registration an explicit owner and teardown path. Cancel timers and detach callbacks; cancellation alone does not release event subscriptions.
- For CTk global scroll/Shift handlers, use a maintained compatibility adapter or a verified dependency fix. Prefer one root-owned dispatcher with weakly held live targets if replacing per-instance global handlers. Preserve mouse, trackpad, Shift-scroll, and nested scrolling semantics.
- Remove only callbacks owned by the retiring widget. Never use blanket `unbind_all` or remove another live widget's bindings. Account for CTk's bind redirection and limitations on selective unbinding rather than assuming normal Tk APIs apply unchanged.
- Detach WrappedLabel's parent callback when that label is destroyed independently; preserve the parent's other Configure handlers. Keep custom lifecycle logic in shared components, not repeated patches across pages.
- Re-run retention probes on the bundled macOS dependencies. Verify bounded bindings, Tcl commands, images, traces, and scaling/appearance registrations after warm-up. Distinguish legitimate persistent root timers from leaked instance callbacks.

### B. Shared surface and failure inventory

The rows below identify code-backed exposure, not additional reproduced macOS defects. All should be covered by this resolution's tests; change behavior where the stated mechanism is confirmed or a concrete cleanup gap is already demonstrated.

| Surface / source | Verified exposure | Required treatment |
| --- | --- | --- |
| [Input SettingsPanel](src/pipeline_calculator/gui/settings_panel.py) and [file selection layout](src/pipeline_calculator/gui/pages/file_select_page.py) | Settings has a separate overflow adapter. Unlike AutoScrollFrame, it leaves CTk's direct scrollbar position callback in place; its maximum height also changes with parent layout. | Bring scrollbar scheduling and teardown under the shared adapter, retaining content-sized settings behavior. Test compact-mode, parameter-column, and overflow thresholds together. |
| [ModalBody](src/pipeline_calculator/gui/modal.py), [parameters](src/pipeline_calculator/gui/dialogs/params_dialog.py), and [analysis session](src/pipeline_calculator/gui/controllers/analysis_session.py) | Parameters and warning/progress bodies share AutoScrollFrame; warning/progress transitions change content height and visible sections. | Carry the scroll/lifecycle fix through these consumers. Verify fixed actions remain reachable after repeated open/close, warning/continue/cancel, resizing, and result publication. |
| [Corridor dialog](src/pipeline_calculator/gui/dialogs/corridor_dialog.py) and [package export dialog](src/pipeline_calculator/gui/actions/export_actions.py) | Both instantiate raw CTkScrollableFrame, bypassing AutoScrollFrame. Text/progress/error changes alter content height. | Adopt the maintained shared scrolling behavior where applicable, preserving each dialog's sizing. Test close/retry/error/completion and return to Summary, including while worker polling is pending. |
| [Native tables](src/pipeline_calculator/gui/tables.py), [scope selector](src/pipeline_calculator/gui/pages/results_page.py), and [corridor row buttons](src/pipeline_calculator/gui/tabs/overlap_tab.py) | Table creation and scope-bar construction call `theme_use('clam')`. Styles use per-widget IDs. State maps specify some, but not every interaction color; combo popup is a separate native surface. | Initialize the intended ttk theme once per Tcl interpreter and centralize reusable styles. Verify selected/unfocused, readonly, disabled, hovered, pressed, popup, heading, trough, and arrow states. Test that creating another table/dialog does not change existing controls. Measure style-definition growth during scope churn; avoid unbounded per-instance styles where unnecessary. |
| [ResultPages navigation](src/pipeline_calculator/gui/layout.py) | Tabs switch to an option menu based on measured label width; navigation itself changes packing. | Test just below/at/above the measured threshold, repeated same-tab clicks, long labels, and switching while resizing. Keep tab strip, menu value, actual visible page, and keyboard focus consistent. |
| [Scope replacement](src/pipeline_calculator/gui/pages/results_page.py) | `select_scope` destroys existing tabs before creating the replacement and eagerly constructs all populated tabs. Same-scope selection also rebuilds. | Make same-scope requests a no-op. Ensure a construction failure leaves a usable page with an explicit error and recovery path, rather than an unexplained blank area or selected scope that disagrees with visible data. Prefer keeping the old valid view until replacement succeeds; bound temporary memory and clean abandoned builds. |
| [Pipelines](src/pipeline_calculator/gui/tabs/pipelines_tab.py), [Placemarks](src/pipeline_calculator/gui/tabs/placemarks_tab.py), [Diagnostics](src/pipeline_calculator/gui/tabs/diagnostics_tab.py) | Rows are synchronously inserted for every record; pipeline sorting reorders every row on the GUI thread. Scope changes repeat construction. | Measure initial and scoped rendering separately from warm tab returns. If budgets fail, use lazy per-tab population and bounded batches or pagination, with explicit progress and generation/cancellation guards. No Tk calls from workers, no stale rows after a scope change, and no silent truncation. |
| [CorridorTable](src/pipeline_calculator/gui/tabs/overlap_tab.py) and [HeaderSorter](src/pipeline_calculator/gui/sorting.py) | Row buttons are placed over Treeview cells after scroll/configure events; sorting changes row identity/order; positioning reapplies styles and sizing. | Check return after horizontal/vertical scroll, sorting, paging, resize, and scope changes. Hidden/retired tables must not keep doing positioning work. Visible buttons must target the correct row and never cover headings, neighboring cells, or scrollbars; disabled maps remain disabled. |
| [Parameter callbacks](src/pipeline_calculator/gui/dialogs/params_dialog.py) and [export actions](src/pipeline_calculator/gui/actions/export_actions.py) | Parameter apply/cancel callbacks catch broad exceptions silently. Ordinary XLSX/JSON export writes synchronously after Save As; state-package export uses a worker. | Surface unexpected transition failures without losing the current results. Profile ordinary export as a distinct full-event-loop stall candidate and move expensive work off Tk if needed, using existing job patterns. Preserve export contents and error recovery. |

Theme scope remains the application's explicitly dark UI, as set in [AppWindow](src/pipeline_calculator/gui/window.py). Testing both macOS system appearances checks compatibility; it does not add an application light-theme feature. Do not replace every native widget indiscriminately: native dialogs can retain their OS styling when readable and functional.

## Consolidated implementation order

The original five steps remain applicable; expand their delivery order as follows:

1. Establish the stable feature baseline, retention probes, and timing measurements; gather artifact identity and a failing macOS reproduction as access permits. Keep a source-to-artifact record. Build-script feature changes are committed in `a8640fd` and must be preserved.
2. Fix the known disclosure rendering dependency and any demonstrated ttk state-style defects, consolidating shared styles where needed. Review all listed interactive states before considering contrast work complete.
3. Fix demonstrated subscription retention, then unify scroll scheduling/lifecycle behavior across Summary, settings, modal bodies, and raw scroll-frame consumers. Retest every consumer whenever the adapter changes.
4. Make tab/scope transitions idempotent, layout converge, and rendering failures recoverable. Preserve current result snapshots and valid focus/scroll state. Do not expand the fix into a persistent all-scope widget cache that retains large views indefinitely.
5. Address measured bulk-table/sort/export latency using bounded work. A lazy renderer must show readable loading/error/empty states and must never leave a selected page blank while waiting. Coalesce repeated requests and discard stale result-generation callbacks.
6. Run the complete shared-surface regression suite and final installed-DMG checks below. Deliver one evidence report mapping each inventory row to a passing scenario or a documented, measured reason no change was needed.

## Expanded regression matrix and measurable completion

Use small deterministic fixtures for lifecycle correctness and larger presentation fixtures for performance; UI fixtures need not repeatedly run costly geometry analysis. Include an end-to-end import of the reported KMZ separately. Record pipeline, placemark, diagnostic, overlap-section, and state counts with every benchmark.

| Test family | Cases and assertions |
| --- | --- |
| Content states | Empty/one-pipeline/no-overlap results; zero versus unavailable estimates; incomplete results; many diagnostics; long file/pipeline/state labels and Unicode; shared-only state, unavailable state overlap, and no state geometry. An empty result must have readable explanatory content. |
| Layout convergence | Width sweeps around card, tab/menu, compact-settings, parameter-column, and scrollbar thresholds, plus short/high-DPI windows. Resize hidden views, then reveal. After user input stops, layout callbacks must settle within a declared bounded interval (target 500 ms on reference hardware) and remain quiet except documented root polling. |
| Navigation and focus | At least 100 ordinary tab-return cycles and 100 scope/dialog replacement cycles; same selection repeatedly; rapid queued selections; scope lacking the previously selected tab. Check page/selector agreement, keyboard focus on a live visible control, disclosure state, valid scroll offset, and correct result generation. Exercise actual button/menu/keyboard events as well as direct method calls. |
| Nested scrolling | Trackpad and mouse wheel over card text, controls, Summary state table, modal body, and scrollbar; horizontal scrolling and Shift press/release; movement between windows while Shift is down. Only the intended live surface should scroll. Explicitly define edge behavior for nested table/outer Summary scrolling and verify no double movement. |
| Lifetime and recovery | Open/close/reopen parameters and both export/corridor paths; cancel analysis; close while callbacks are pending; repeat import/reanalysis. Verify no retained instance growth, no invalid Tcl command/background errors, no stale worker updates, no orphaned modal grab/overlay, and no steadily increasing event-dispatch cost. Use weak references and actual Tcl binding/command counts in addition to memory sampling. |
| Injected failures | Raise a controlled exception while constructing a scope/tab, starting a worker, applying parameters, or rendering a batch; delay completion then change scope/import. Show an actionable error, keep or restore a usable view, clear busy state appropriately, and preserve the analysis snapshot. Stub external corridor launches and file dialogs in automated tests. |
| Visual state transitions | All control states in the inventory, including selection after focus moves elsewhere, scope popup open/close, disabled corridor action, preference notices appearing/disappearing, warning-to-progress transitions, and export error-to-retry. Capture real pixels; property values alone cannot prove contrast. |
| Platform and packaging | Both supported entrypoints on Windows and packaged macOS; supported architectures for which artifacts ship; standard/Retina scaling and moving between displays where available; minimize/restore, app deactivation/reactivation, and modal return. Record exact runtime patch versions rather than assuming `TkVersion >= 8.6` establishes identical rendering behavior. |

Performance measurement rules:

- Keep the earlier 100 ms p95 / 250 ms maximum warm Summary-return targets. Distinguish initial construction, first visit to a lazy tab, scope replacement, sorting, and export. For expensive operations, target visible feedback within 100 ms and continued heartbeat service within 250 ms, with operation completion times reported separately.
- Measure after deliberate input, with a documented warm-up and representative hardware. Exclude time spent waiting for the user in a native file dialog or while the application is suspended/minimized from active-work latency statistics; assess restoration separately.
- Native widget mapping/content geometry is an automated proxy for visibility, not proof of actual screen paint. Pair it with macOS screen recordings or screenshots timed to interactions. Keep an external watchdog because a blocked Tk loop cannot run its own timeout callback.
- Capture Python callback exceptions **and Tcl background errors/stderr**. Preserve stack samples and a bounded recent event trace on timeout. Do not accept a swallowed exception, a sleep added to every switch, or an empty-but-mapped frame as success.
- For leak checks, establish a post-warm-up baseline and verify counts plateau across multiple batches, allowing documented interpreter-owned persistent resources. Do not demand total process RSS return exactly to its original value after every destruction.

Retain the regression scenarios in the normal test suite, with a small deterministic cross-platform subset on every build and the longer repetition/performance/visual checks as a documented release gate. Reuse [native GUI isolation](tests/conftest.py), [layout probes](tests/ui_layout_probe.py), [state UI tests](tests/test_state_breakdown_ui.py), and [packaged smoke validation](scripts/validation/check_packaged_smoke.py); extend their assertions rather than creating competing test frameworks. Verify which scenarios actually ran on macOS instead of treating skipped Windows-only tests as coverage.

This audit adds two measured cleanup gaps and a broader, source-backed regression scope. It does not claim that every listed risk is an existing user-visible bug, or that Windows probes establish the cause of the original macOS blank Summary.
