# Dark mode, DPI and layout validation

September 10, 2026. **Implementation, regression checks and Windows packaging validation complete.**

## Changes

- Both entrypoints now share a dark CustomTkinter root with the existing TkDND integration. This fixes the white client background and enables the framework's Windows dark title bar and per-monitor DPI tracking. Theme selection happens before widgets are constructed.
- Startup sizing uses the nearest monitor's Windows work area, excluding the taskbar, converts physical pixels to logical dimensions, reserves room for native window borders, and clamps the initial size. Monitor/work-area/DPI changes trigger a delayed refit after CustomTkinter settles its sizing constraints. Returning to results preserves the user's window size.
- File selection and parameter editing use scrollable forms. Hints wrap; inputs remain aligned. Bottom actions reserve their own space and reflow into rows instead of overlapping. Results use a compact section selector; a read-only, horizontally navigable filename field prevents long names from consuming the viewport.
- All four tables have dark headings, selection and scrollbars, both scroll axes, and DPI-scaled rows, fonts, columns and scrollbar arrows. Long cell values remain reachable horizontally.
- Corridor actions use a selected row and a normal View Corridor button, Enter, double-click, or the Action cell. Floating buttons over native table cells were removed. Pagination preserves the original global section index. Pairwise mileage and the approximate-corridor explanation remain in Summary.
- The corridor dialog fits the monitor, scrolls long error messages and keeps its path and actions reachable. Analysis warnings appear before filenames, wrap in logical DPI units, and retain accessible Cancel/Continue controls.
- Legacy and modular entrypoints share the same pages, parameter dialog and tables. Pipeline calculations and corridor geometry algorithms were not changed.

## Direct verification

- Full regression suite: **238 tests passed** after the normal-startup correction (39.58 seconds).
- Twenty work-area cases cover 1024/1920/2560/3840 pixel monitor layouts, negative monitor origins and scaling factors 100%, 125%, 150%, 200% and 250%.
- Six isolated native Windows probes exercise all result sections, import/retry, parameter editing, long filenames and diagnostics, corridor recovery and the workload caution. Logical size/scaling cases: 440x340 at 100%, 640x480 at 125%, 800x600 at 150%, 480x320 at 200%, 1000x720 at 250%, plus the legacy entrypoint at 640x480/125%. Native minimum-size constraints may enlarge a requested test window.
- Probes assert dark native titlebar state using DwmGetWindowAttribute, TkDND availability, visible tables, both scroll axes, button text fit and button bounds, and global corridor indices 21/41 after pagination. Callback exceptions fail the probes. Each process has a timeout so a resize loop fails verification.
- Actual app-window captures inspected at compact 100%, large 125%, and compact 200% layouts. Local captures live in `.validation-output/visual-polish/`; capture commands show only the test app, and do not change OS display settings.
- Verification found and fixed a wrapping callback loop, stale grid column allocation after button reflow, and a table collapsed by surrounding controls at small sizes.

Reproduce with `.venv/Scripts/python.exe -m pytest -q`. For local visual captures run `.venv/Scripts/python.exe tests/ui_layout_probe.py 2 480 320 .validation-output/visual-polish` on Windows.

Source commit: `ccdc1f12ba3bc4a26a9f8a15175810c31ed6243d` plus working-tree changes.
Python source fingerprint: `473c5752897165bdd3a3bc219b40d526506c40843067686fb62cb4b7c5f04787`.

## Scope and deferred checks

This is verified Windows behavior over the matrix above, not a guarantee for every hardware/OS configuration. Scaling is injected through the real CustomTkinter DPI path in isolated processes; tests do not change the user's display settings. Physical mixed-DPI monitor transfers, RDP/docking changes, unusual accessibility settings, and macOS rendering still need hardware/platform acceptance in the follow-up runbook. Extremely small usable work areas below the tested window sizes are not certified. Native OS file pickers/message boxes follow Windows settings; custom app pages and dialogs are dark.

Primary references: [CustomTkinter scaling](https://customtkinter.tomschimansky.com/documentation/scaling/), [MonitorFromWindow](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-monitorfromwindow). The installed CustomTkinter 5.2.2 window/scaling implementation and TkinterDnD loader were also inspected directly.


## Completed Windows artifacts

Both modular and legacy packages passed the frozen calculation/export/TkDND smoke
check at version `4.5-dev.ccdc1f12ba3b.dirty`. The modular executable was additionally
launched in normal app mode: its real main window appeared, the native dark titlebar
attribute was enabled, and its window rectangle was `(516, 28, 1530, 786)` on this host.
The test window was closed after verification. No application code was committed or
release published; `.dirty` accurately identifies a local working-tree build.

The refreshed modular build is available as `dist/Pipeline_Calculator_v4.exe` and
`dist/Pipeline_Calculator_v4.5-dev.ccdc1f12ba3b.dirty.exe`. The prior alias was preserved
under `dist/archive/` with its SHA-256 prefix. Copied artifact hashes match the tested
build. Local records: `.validation-output/visual-polish-build.json`,
`.validation-output/visual-polish-source.json`, `.validation-output/visual-polish-tests.txt`.

Executable SHA-256: `4AA141FFB0237934B741A6AADE41C470087EA3C258BBC554A1192ED40240CC21`.


## Normal-startup correction (September 10, 2026)

The owner reproduced a startup regression in the first visual-polish EXE: the
window appeared briefly and disappeared, leaving its process running. Direct
inspection confirmed hidden main windows, and a source run reported
`state=withdrawn`, `winfo_viewable=0` after entering the real mainloop.

Cause: AppWindow called `withdraw()` before CustomTkinter's first mainloop. CTk
5.2.2 retains a pre-mainloop withdrawal flag that `deiconify()` does not clear.
Its mainloop titlebar refresh then hides the window. Removed that redundant call;
CustomTkinter now controls initial visibility itself.

The prior layout probes called `update()` before the mainloop, and the packaged
window inspection explicitly showed a hidden window. Those checks were insufficient
to validate ordinary startup; the earlier packaged-window observation must not be
read as proof that an Explorer launch remained visible.

Two new isolated tests enter the real mainloop directly (modular and legacy) and
assert normal, visible state at 350 and 1600 milliseconds. All 238 tests pass.
The rebuilt modular EXE passes calculation/export packaging smoke and the new
`scripts/validation/check_windows_startup.py` check: ordinary subprocess launch,
no forced visibility, at least five seconds continuously visible. Both the build
output and the exact `dist/Pipeline_Calculator_v4.exe` alias pass this check.
The stale hidden startup instances were closed, and the faulty alias was archived
before replacement. Current evidence: `.validation-output/startup-fix-build.json`,
`startup-fix-tests.txt`, `startup-fix-normal-launch.json`, and
`startup-fix-dist-launch.json` in the same local evidence directory.


## Import screen refinement (September 10, 2026)

Owner-requested refinement complete: a blue-outlined KMZ/KML drop box with Browse
inside it appears above a separate Analysis Settings section. Settings use two
compact columns when space permits and a single scrollable column on narrow
windows. The drop box stays visible while settings scroll. Both entrypoints share
the layout. Dropped paths use Tcl list parsing and reject multiple files clearly.

Verified the large-window capture and existing scale/size matrix: 55 focused tests
passed, followed by 28 final layout/startup tests after a small callback cleanup.
The rebuilt Windows EXE passed the ordinary-launch visibility check and replaced
`dist/Pipeline_Calculator_v4.exe`; the previous EXE is archived. The alias hash
matches the tested build. Evidence: `.validation-output/import-layout-build.json`,
`import-layout-final-tests.txt`, `import-layout-startup.json`, and captures in
`.validation-output/import-layout/`.


## Bottom-anchored settings (September 10, 2026)

Moved the compact settings panel to the bottom of the import screen, with a
240-logical-pixel preferred height. Extra vertical space now sits between the
file drop area and settings. Small windows retain the settings scrollbar.
The visual capture and browser preview were refreshed. Native layout/startup
checks were repeated, and the rebuilt Windows executable passed the sustained
normal-launch check before replacing the v4 alias. The previous build is archived.
Evidence is recorded in `.validation-output/bottom-settings-build.json`,
`bottom-settings-tests.txt`, and `bottom-settings-startup.json`.


## Separate middle drop field (September 10, 2026)

Kept Browse at the top with file-picker instructions and placed a dedicated,
outlined drag-and-drop field in the middle space. Its own centered instructions
explain supported files and that dropping starts analysis. Settings remain below.
The actual drop registration now targets the middle field. The refreshed capture
matches this layout, and the local browser preview was updated. All 28 focused
layout/startup checks passed; the rebuilt EXE passed normal-launch verification
and replaced the v4 alias, with matching hashes and the old build archived.
Evidence: `.validation-output/middle-drop-build.json`, `middle-drop-tests.txt`,
`middle-drop-startup.json`, and captures in `.validation-output/middle-drop/`.


## Gray dotted drop field (September 10, 2026)

Updated only the middle drag-and-drop field: lighter gray (#3A3A3A) interior,
clean rectangular edge, and dotted blue (#5FA5D5) outline. The outline scales
with DPI and follows resizing. Its canvas is registered as a file drop target;
existing instructions and drop handling remain. Browse and settings styling are
unchanged. Inspected the rendered capture, passed 28 layout/startup tests and
verified the rebuilt EXE stays visible on normal launch. Updated the v4 alias
and browser preview; archived the prior build. Evidence is in
`.validation-output/gray-drop-build.json`, `gray-drop-tests.txt`,
`gray-drop-startup.json`, and `.validation-output/gray-drop/`.


## Settings edges, spacing and automatic scrollbar (September 10, 2026)

The bottom settings panel now spans its parent edge to edge with no exterior
bottom padding. Its viewport fits the content up to a 240-logical-pixel cap,
removing the empty band below the cards. The scrollbar is removed when content
fits and restored for overflow; resizing and content layout changes update it.
The CTk viewport/scrollbar integration is isolated in `gui/settings_panel.py`.
Very short windows collapse redundant Browse headings so settings remain
reachable, including after returning from results. Browse itself stays available.

Passed all 28 layout/startup checks, including new assertions for scrollbar
visibility matching overflow, full settings width and absence of extra viewport
height. Inspected large and compact native captures. The startup probe now allows
CTk's transient initial titlebar redraw but still requires a visible settled
mainloop; the packaged check requires sustained visible ordinary startup.
The rebuilt EXE passed that check, replaced the v4 alias with a matching hash,
and the prior version was archived. Browser preview refreshed. Evidence:
`.validation-output/settings-fit-build.json`, `settings-fit-tests.txt`,
`settings-fit-startup.json` and captures in `settings-fit/` and `settings-fit-small/`.
