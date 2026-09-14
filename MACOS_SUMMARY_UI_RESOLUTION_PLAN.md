# macOS Summary UI resolution plan

Status: investigated; implementation deferred to avoid concurrent edits.
Date: 2026-09-14. Inspected repository HEAD: `212d0d7` plus the active, uncommitted state-breakdown feature.

## Scope and handoff

Resolve two reported DMG defects: the Additional details disclosure becomes a white bar after interaction, and Summary can remain blank or take too long to display after visiting other tabs.

The active task **Plan state-aware mileage toggle** is using this same checkout. Its modified files include `summary_tab.py`, `layout.py`, and `results_page.py`, which are directly involved here. This investigation added only this plan; it did not edit implementation, tests, dependencies, or build files. Do not restore or replace those files with an older version. Implement against the completed feature, preserving state selection and combined/state accounting.

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

## Implementation sequence after the feature settles

1. **Identify and reproduce the binary.** Record artifact URL/run ID, commit, app version, macOS version, architecture, display scale, appearance, and bundled Python/Tcl/Tk/CustomTkinter versions. Reproduce with the user's KMZ when available, plus small and large fixtures. Record whether expanding/collapsing or scrolling is necessary. Compare the reported build with the completed feature baseline.

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
