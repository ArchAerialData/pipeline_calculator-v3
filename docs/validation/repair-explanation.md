# Plain-language repair details

Verified on Windows, 2026-09-18.

Expanded repair details now show a concise **What this means** explanation above the separately labeled **Technical details** text box. Both sections collapse with Hide details. The body wraps and scrolls at constrained sizes; action buttons remain outside it. The normal expanded dialog fits without an unnecessary outer scrollbar in the reviewed 150% capture.

The explanation maps the four implemented repair rules to plain language: missing formatting declaration, descriptive-text ampersand escaping, leading header whitespace, and a filename/type mismatch. Eligible repairs describe proposed changes; verified reports describe completed changes. Unknown rules refer to the technical report instead of guessing a fix. Failure guidance distinguishes client-source issues from verification failures, operational problems, and inspection limits. Existing full technical reports and Copy client request content are preserved.

The change is presentation only. [Source manifest](repair-explanation/source-manifest.json) confirms that only `gui/repair_messages.py` and `gui/repair_ui.py` differ from the previous title-layout build; parser, geometry, mileage and export source files are unchanged.

## Verification

```powershell
.venv/Scripts/python.exe -m pytest tests/test_repair_messages.py tests/test_repair_ui.py -q
```

**14 passed in 23.82 seconds.** [Output](repair-explanation/tests.txt). Coverage includes eligible versus verified wording, unknown/multiple rules, source versus application failures, preserved client requests, repeated detail expansion/collapse, readable summary widths and order, high DPI, long filenames, keyboard actions, cleanup, and both GUI entrypoints. `git diff --check` passed.

Reviewed isolated native screenshots with synthetic data:

- [Expanded explanation at 150%](repair-explanation/screenshots/offer-expanded-900x650-150pct.png)
- [390-pixel narrow window](repair-explanation/screenshots/offer-expanded-390x650-100pct.png)
- [Unsafe-repair failure](repair-explanation/screenshots/failure-900x650-100pct.png)

The Windows executable is built in an isolated staging directory and published only after both entrypoints pass frozen offline smoke checks. [Build output](repair-explanation/windows-build.txt), [smoke results](repair-explanation/windows-smoke/summary.json), and [publication/hash receipt](repair-explanation/windows-publication.json) document the final artifact. Existing running apps are preserved; reopen the local executable to load the change. This does not constitute macOS verification or a remote release.
