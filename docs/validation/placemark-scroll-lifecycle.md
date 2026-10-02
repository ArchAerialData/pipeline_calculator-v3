# Scroll viewport teardown regression

October 2, 2026. Follow-up to the [folder overview](placemark-folder-overview.md).

## Report and direct reproduction

The [macOS CI job](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/37043349264/job/110958608078)
for `74c36e0cde634bd3d54b1017dea3a8957bf88fee` failed
`test_long_names_duplicate_folders_and_unrecorded_or_empty_inventories`.
The fourth replacement view's total card measured 1 pixel wide. The supplied
log reports **1 failed, 1,052 passed, 48 skipped** on Python 3.11.9 / macOS.

Direct inspection of the installed CustomTkinter 5.2.2 implementation showed
that `CTkScrollableFrame.pack/grid` lays out an outer frame, while its `destroy`
method destroys only the inner content frame. Destroyed views therefore left
their wrapper, canvas, and scrollbar occupying the parent layout. Replacing
the view repeatedly eventually left no usable viewport for the new content.

This was reproduced on Windows before changing production code:

- The strengthened placemark test failed on its first teardown because
  `wrapper.winfo_exists()` still returned 1 (**1 failed**).
- New shared lifecycle tests failed for direct teardown with both `pack` and
  `grid` (**2 failed, 4 passed**). Host-first and wrapper-first cases passed.

The original card-width assertion alone missed the leaked Windows viewport;
the new checks verify native viewport visibility and actual widget removal.

## Correction

[AutoScrollFrame.destroy](../../src/pipeline_calculator/gui/scrolling.py) now
retires the content and then destroys its enclosing frame. Content goes first
so the canvas child list no longer contains the view during wrapper teardown.
The disposed guard makes repeated teardown safe. Existing timer cancellation
and owned global binding removal still run before either widget is destroyed.

The [placemark regression](../../tests/test_placemark_cards_ui.py) checks each
replacement's visible viewport and restoration of the parent's original layout
children. The [shared lifecycle tests](../../tests/test_ui_lifecycle.py) cover
five successive replacements per layout manager, direct and ancestor teardown,
repeated destruction, timer/binding removal, and unrelated sibling preservation.
The [Summary regression](../../tests/test_scroll_viewport.py) now queues a native
layout notification before teardown and requires the whole viewport to disappear.
Card size/wrapping assertions and existing wait durations remain unchanged.

## Validation

Host: Windows 10 build 19045, Python 3.11.9, CustomTkinter 5.2.2, Tk 8.6.
Native tests run in the repository's isolated GUI subprocess harness.

- Corrected placemark regression: **1 passed** (5.90 seconds).
- Six new shared lifecycle cases: **6 passed** (8.12 seconds).
- Broader UI/layout run: **74 passed, 2 failed** (373.00 seconds), covering
  `test_placemark_cards_ui.py`, `test_ui_lifecycle.py`, `test_scroll_viewport.py`,
  `test_ui_shared_surfaces.py`, `test_summary_tab.py`, `test_browse_ui.py`,
  `test_state_breakdown_ui.py`, and `test_ui_layout.py`. The folder-card suite
  passed all sixteen size/scaling combinations (100-200%), and the queued
  Summary notification regression passed. Local JUnit evidence is in
  `.validation-output/placemark-scroll-lifecycle.xml`.
- The 250% modern layout probe failed its settings-gap assertion on return to
  Import, then passed unchanged in isolation (**1 passed**, 10.83 seconds).
  This intermittent result is retained rather than counted as an initial pass.
- The 100-cycle scope replacement case exceeded its unchanged 120-second
  deadline in both the broad run and an isolated rerun. Both made steady
  progress through 80 replacements (106.10 and 109.16 seconds respectively).
  The isolated JUnit record is
  `.validation-output/placemark-scroll-lifecycle-recheck.xml`.
- A separate 20-cycle profile measured 18.24 seconds with the fix versus 15.97
  seconds with the original teardown. Total widget destruction increased by
  about 0.34 seconds; Tk drawing/event processing dominated both runs. These
  local timing samples omit the failure-injection/heartbeat paths and ran
  outside the isolated desktop. They are diagnostic, not a substitute for the
  full test or a portable performance guarantee.
- The exact full 100-cycle test was then run in the isolated GUI harness with
  only `AutoScrollFrame.destroy` restored to its original implementation inside
  that child process. It **also timed out** at 120.05 seconds, reaching 90 cycles
  in 113.46 seconds. This reproduces the timeout without the fix. The original
  deadline, heartbeat, injected failure, and assertions were all preserved;
  tracked source was never reverted. The baseline output is retained in
  `.validation-output/scope-baseline-full.txt`. The local stress-test timeout
  remains unresolved and the broader suite is not reported as fully passing.

Native macOS execution is not available on this host. The existing macOS build
runs these portable native tests as part of its full pytest gate. A CI run
containing this fix is required to verify the macOS result; rerunning the old
commit would still test the unfixed code. No application bundle was rebuilt.
