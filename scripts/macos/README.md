# macOS scripts

These scripts are for engineers building/running Pipeline Calculator v4 from source on macOS, and for producing a self-contained `.app` for distribution.

## Development (from source)
- Setup dev machine: `bash scripts/macos/setup_macos.sh`
- Run GUI: `bash scripts/macos/run_gui.sh`
- Basic checks: `bash scripts/macos/run_tests.sh`

These scripts use a repo-local virtualenv at `.venv/`.

## Distribution build (no Python needed on pilot machines)

Build a self-contained `.app` with PyInstaller:
- `bash scripts/macos/build_app.sh`

To build the legacy monolithic GUI instead of the default modular v4 GUI:

```bash
PIPELINE_CALCULATOR_BUILD_IMPL=legacy bash scripts/macos/build_app.sh
```

Then package a DMG:
- `bash scripts/macos/package_dmg.sh`

Note: CI produces **signed but not notarized** DMGs (matches the reference repo behavior). Gatekeeper prompts are expected outside the Mac App Store unless you notarize.

## Notarization (manual, outside GitHub)

If you want to avoid Gatekeeper prompts for pilots, notarize the distribution artifact **after** GitHub produces the signed DMG:

```bash
bash scripts/macos/notarize_dmg.sh dist/Pipeline_Calculator_v4.N.dmg
```

This uses App Store Connect API key credentials from env vars or the gitignored `apple_dev/` helper files (see `CODE_SIGNING.md`).
