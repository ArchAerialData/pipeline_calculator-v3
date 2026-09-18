# Repair workflow layout verification

Verified on Windows on 2026-09-18. This change places persistent input notices beside the state selector when both fit, with a stacked layout on narrower windows. Without state breakdown, the notice occupies the header alone. With neither a selector nor a notice, the header collapses. The shared results page covers both GUI entrypoints.

Repair approval now has a full-width green **Repair & analyze** button below **Cancel** and **Choose another file**. Expanded details scroll while the actions remain reachable. Keyboard traversal follows the visual action order; initial focus remains on Cancel. Failure, details and save panels retain their existing actions.

## Verification

The focused UI regression command passed **60 tests** in 194.72 seconds:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_results_header.py tests/test_repair_ui.py tests/test_state_breakdown_ui.py tests/test_ui_lifecycle.py tests/test_ui_layout.py -q
```

[Retained test output](repair-layout/ui-regression.txt). Coverage includes resizing at 100%, 150%, 200% and 250% scaling; state ON/OFF; scope switching; recognized-file-type, failed/cancelled-analysis and preference-save notices; empty-header collapse; keyboard actions; expanded details; callback/trace cleanup; repeated navigation; and both application entrypoints. A separate native probe also found no header overflow at 512, 390, 360 and 320 logical pixels. `git diff --check` passed.

Twelve isolated-window captures use synthetic data, not a client file. Their receipts record actual window sizes; Windows constrained the physical height at 150% scaling. Reviewed examples:

- [Wide state-aware results](repair-layout/screenshots/results-on-1200x760-100pct.png)
- [Narrow state-aware results](repair-layout/screenshots/results-on-620x760-100pct.png)
- [Results without state breakdown](repair-layout/screenshots/results-off-1200x760-100pct.png)
- [Repair approval](repair-layout/screenshots/offer-900x650-100pct.png)
- [Expanded repair details at 150%](repair-layout/screenshots/offer-expanded-900x650-150pct.png)
- [Unsafe-repair failure](repair-layout/screenshots/failure-900x650-100pct.png)

The [source manifest](repair-layout/source-manifest.json) records the candidate and its differences from the prior verified corridor build: only `gui/results_header.py`, `gui/pages/results_page.py`, and `gui/repair_ui.py`. Calculation, geometry, parser and export source files remain byte-identical to that build.

## Local executable

The executable was built into an isolated staging directory, passed offline smoke checks using both GUI entrypoints, and was published to both local `dist` executable paths. Version: `4.24-dev.9aa8698da150.dirty`. SHA256: `37E314A8CFB2CEB4ACBB0BE68A6BD745B53196F38B76CF9129C1272F2A8AFCD6`. Both installed copies match the tested binary. Final destinations and preserved previous builds are recorded in the [publication receipt](repair-layout/windows-publication.json). See the [build log](repair-layout/windows-build.txt) and [frozen smoke summary](repair-layout/windows-smoke/summary.json).

Existing running applications were not closed. Reopen `dist/Pipeline_Calculator_v4.exe` to load the new UI.

This is local Windows verification. No macOS validation or remote release is implied.
