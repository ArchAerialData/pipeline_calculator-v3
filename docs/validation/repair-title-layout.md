# Repair dialog title-row refinement

2026-09-18 follow-up to the [repair layout verification](repair-layout.md).

**Show details / Hide details** now occupies the right side of the repair dialog's title row. The filename and explanation follow underneath; the collapsed explanation is directly above the action area, with existing body and footer padding separating them. On narrow windows the details control moves to the right beneath the wrapped title. Expanded diagnostic text remains in the scrolling body. Dialogs without details leave no empty button column.

Only `gui/repair_ui.py` changed from the preceding tested build; see the [source manifest](repair-title-layout/source-manifest.json). No calculation, geometry, parser or export code changed.

Verification:

- `pytest tests/test_repair_ui.py -q`: **8 passed in 25.81 seconds**, including high DPI, long filenames, expanded details, keyboard actions, both entrypoints and workflow cleanup. [Output](repair-title-layout/tests.txt).
- `git diff --check`: passed.
- Isolated native screenshots with synthetic data were reviewed: [normal width](repair-title-layout/screenshots/offer-900x650-100pct.png), [390-pixel window](repair-title-layout/screenshots/offer-390x650-100pct.png), and [expanded details at 150%](repair-title-layout/screenshots/offer-expanded-900x650-150pct.png). No callback errors; text and actions fit.

The local Windows build and publication evidence are retained in the [build log](repair-title-layout/windows-build.txt), [offline packaged smoke summary](repair-title-layout/windows-smoke/summary.json), and [publication receipt](repair-title-layout/windows-publication.json). Both published executable copies must match the tested binary. Existing running instances are preserved and need reopening to load this refinement.
