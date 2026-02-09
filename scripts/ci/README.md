# GitHub Actions CI (macOS)

This folder contains CI scripts used by the GitHub Actions workflow. The workflow file itself must live under `.github/workflows/` (GitHub requirement), but build/sign/package logic is kept here to reduce workflow clutter.

## Scripts
- `macos_build.sh` — creates `.venv`, installs deps, validates imports, builds the `.app`.
- `macos_package.sh` — packages the `.app` into a `.dmg`.
- `macos_sign_and_package.sh` — signs the `.app` and creates a DMG containing the signed app (no notarization; only runs if secrets are provided).

## Secrets used by CI (optional for code signing)
- `MACOS_CERT_P12` (base64 of Developer ID Application `.p12`)
- `MACOS_CERT_PASSWORD`

## Notes
- Unsigned DMG artifacts are allowed only for non-`main` and non-tag builds (PR/debug builds).
- This repo is configured to produce **signed but not notarized** DMGs (mirrors the reference repo setup). Gatekeeper will typically require a manual override on first launch for apps distributed outside the Mac App Store.
