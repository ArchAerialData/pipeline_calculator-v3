# Code Signing (macOS) for GitHub Actions

This repo is configured to produce a **code-signed macOS `.app` inside a `.dmg`** via GitHub Actions (no notarization).

## What’s signed
- The workflow signs the built `.app` with a **Developer ID Application** certificate, then packages it into `dist/Pipeline_Calculator_v3.dmg`.
- Notarization is **not** performed (matches the reference repo setup). Gatekeeper prompts are expected for non-notarized apps.

## Required GitHub Secrets
Set these as **Repository secrets** (Settings -> Secrets and variables -> Actions):

- `MACOS_CERT_P12`
  - Base64-encoded `.p12` that contains your Developer ID Application certificate + private key.
- `MACOS_CERT_PASSWORD`
  - The password used when exporting the `.p12`.

The workflow will **fail on `main` and tags** if these secrets are missing.

## Local helper (recommended)

This repo uses a gitignored folder `apple_dev/` to stage signing material locally (never commit).

If you have the base64 files staged locally (for example in `apple_dev/`):
- `apple_dev/MACOS_CERT_P12.base64.txt`

You can push secrets to GitHub with the GitHub CLI:

```bash
bash scripts/ci/set_github_secrets.sh owner/repo
```

Notes:
- Requires `gh` installed and authenticated (`gh auth login`).
- The script prompts for `MACOS_CERT_PASSWORD` (or you can pass it via env var).

## Where the workflow lives
- Workflow: `.github/workflows/build.yaml`
- Signing implementation: `scripts/ci/macos_sign_and_package.sh`

