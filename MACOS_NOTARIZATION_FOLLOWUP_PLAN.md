# macOS notarization and credential reuse — follow-up plan

Created: **2026-09-14** (America/Chicago)  
Last updated: **2026-09-14**  
Overall status: **DEFERRED — investigation complete; implementation not started**  
Completion: **planning only; no credentials or application/build code changed**

## Objective

Distribute Pipeline Calculator internally through Dropbox as a signed, notarized
DMG. Reuse the existing Apple developer identity and suitable notarization
credentials from Purway Geotagger rather than repeat the original account setup.
Each new distribution artifact must receive its own notarization approval; an
approval for Purway cannot be inherited by Pipeline Calculator.

The intended coworker workflow is: download DMG, open it, drag the app into
Applications, eject the disk image, and launch from Applications.

## Verified baseline — 2026-09-14

| Item | Status | Evidence / limitation |
| --- | --- | --- |
| Pipeline Calculator builds and code signing | COMPLETE | September 14 merged build succeeded; existing workflow signs the app and packages a DMG. Signing is separate from notarization. |
| Pipeline signing secrets | VERIFIED | Repository contains `MACOS_CERT_P12` and `MACOS_CERT_PASSWORD`. Values were not read or compared against Purway. |
| Purway notarization credential names | VERIFIED | Repository contains `APPLE_API_KEY_P8`, `APPLE_KEY_ID`, and `APPLE_ISSUER_ID`, alongside its signing secrets. Presence does not establish validity or current Apple permissions. |
| Purway automated notarization | DISABLED | Current workflow calls `macos_sign_and_package.sh`. The former CI `macos_sign_and_notarize.sh` redirects to signing/package and explicitly states notarization is disabled. |
| Purway local notarization script | AVAILABLE, NOT VALIDATED | `scripts/macos/sign_and_notarize.sh` uses a Mac keychain profile. Its existence does not establish that the profile or credentials are available on this Windows computer or on CI. |
| Pipeline automated notarization | NOT IMPLEMENTED | Current signing/package script explicitly reports that the artifact is not notarized. |
| Automatic secret inheritance | NOT AVAILABLE IN CURRENT SETUP | Both repositories belong to the personal GitHub account `ArchAerialData`. Repository secrets do not automatically carry across repos. |
| Original Apple private-key file availability | UNKNOWN | No search for private-key contents or credential transfer was performed. GitHub's secrets API exposes metadata, not saved plaintext values. |

Repositories:

- Pipeline Calculator: `ArchAerialData/pipeline_calculator-v3`.
- Purway Geotagger: `ArchAerialData/purway_geotagger_app`.
- Local Purway checkout: `C:\Users\rbake\Desktop\VS Code Shortcuts\purway_geotagger_app`.

## Implementation checklist

Status meanings: **COMPLETE** = verified work finished; **DEFERRED** = intentionally
not started; **NEEDS INPUT** = specific missing material or decision; **PENDING
VERIFICATION** = implemented but not yet proven. Update dates and evidence as work
progresses; do not mark credential presence or a green signing job as notarization.

### 1. Reconfirm the baseline

Status: **DEFERRED**

- [ ] Read applicable repository instructions and current workflow/scripts.
- [ ] Check whether another task has already enabled notarization or changed secrets.
- [ ] Inspect the latest successful Pipeline and Purway builds without printing secrets.
- [ ] Confirm that the existing signing identity and candidate notarization key belong
  to the intended Apple developer team and remain usable.

### 2. Make notarization credentials available

Status: **NEEDS INPUT — availability of the original Apple key is unknown**

- [ ] Determine whether the original `.p8` key and matching key/issuer IDs are
  available in an approved local or secure storage location. Do not request that
  private-key contents be pasted into chat or committed to either repository.
- [ ] If available and still valid, configure the three matching repository secrets
  in Pipeline Calculator using a secure input mechanism.
- [ ] Keep Pipeline Calculator's already-working signing certificate and password;
  no replacement certificate is currently indicated.
- [ ] If the key exists only as a GitHub secret, document the limitation and choose
  an authorized approach: obtain a replacement suitable API key, or separately
  design a shared signing service/workflow with explicit access controls. Do not
  expose a saved secret through build logs to recover it.

Do not assume organization-level shared secrets are available under the current
personal-account ownership. A migration to an organization or a shared signing
service is a separate scope decision, not a prerequisite for the simplest fix.

Completion evidence: required secrets configured and a successful Apple
authentication check, recorded without secret values.

### 3. Implement Pipeline Calculator notarization

Status: **DEFERRED — depends on step 2 for live verification**

- [ ] Preserve the app's own bundle identifier and existing signing setup.
- [ ] Adapt the relevant Purway pattern for Pipeline Calculator's artifact paths;
  do not copy its app name, bundle identifier, or local keychain-profile assumption.
- [ ] Add Apple API-key authentication to the CI notarization step. Decode private
  material into temporary files with restricted permissions and remove it on exit.
- [ ] Submit the signed distribution container with `notarytool`, wait with a
  bounded timeout, and require an explicit **Accepted** result.
- [ ] Preserve actionable notarization status/logs on rejection or timeout, without
  exposing credentials. A rejected or timed-out submission must fail distribution.
- [ ] Staple and validate the ticket on the final DMG. If the standalone app remains
  a distribution artifact, staple/validate it too and package the final app correctly.
- [ ] Retain signature verification and add appropriate Gatekeeper assessment.
- [ ] Enforce required credentials on distribution builds; avoid silently publishing
  signed-only output as a notarized release. Keep untrusted PRs away from secrets.
- [ ] Add the resulting acceptance status, artifact identity, and verification
  results to CI evidence and documentation.

Scope: change Pipeline Calculator only. Re-enabling Purway's automated notarization
requires its own follow-up scope; do not modify that application's production
workflow as an incidental part of this task.

### 4. Verify and prepare distribution

Status: **DEFERRED — automation follows step 3; interactive acceptance remains separate**

- [ ] Test missing credentials, Apple rejection, timeout, and stapling failure:
  each must fail clearly and prevent a misleading distribution artifact.
- [ ] Complete a real macOS Actions build with Apple acceptance, signature checks,
  stapling validation, and retained diagnostic reports.
- [ ] Confirm Windows builds remain successful.
- [ ] Check the actual DMG contents and provide a clear Applications-copy workflow.
- [ ] Confirm supported CPU architecture and minimum macOS version from the build;
  do not assume Intel support from an Apple Silicon build.
- [ ] In the deferred manual acceptance session, download the DMG through Dropbox
  on a coworker-like Mac, copy the app to Applications, and verify normal launch
  without Terminal commands or security-setting overrides.
- [ ] Verify a representative calculation after installation.
- [ ] Publish the DMG and concise installation instructions through the approved
  internal distribution process. Upload the DMG itself, not GitHub's outer ZIP.

## Resume instructions and completion record

Resume with step 1. Resolve only the missing credential availability/decision with
the owner; routine script adaptation, automated tests, CI diagnosis, and docs can
proceed autonomously once the necessary material and scope are available. Keep
interactive Mac and real-project testing together with the existing follow-up
acceptance session rather than requesting piecemeal tests.

| Milestone | Status | Completion date / evidence |
| --- | --- | --- |
| Initial investigation and this plan | COMPLETE | 2026-09-14; sources below |
| Credential reuse / configuration | NEEDS INPUT | Not started |
| Pipeline notarization implementation | DEFERRED | Not started |
| Real Apple acceptance and stapling | DEFERRED | Not verified |
| Dropbox download/install acceptance | DEFERRED | Not verified |
| Internal distribution ready | DEFERRED | Requires preceding verification |

## Sources

- [Pipeline September 14 successful merged build](https://github.com/ArchAerialData/pipeline_calculator-v3/actions/runs/34854365853).
- [Pipeline signing/package script](scripts/ci/macos_sign_and_package.sh).
- [Purway current macOS workflow](https://github.com/ArchAerialData/purway_geotagger_app/blob/main/.github/workflows/macos-build.yml).
- [Purway deprecated CI notarization script](https://github.com/ArchAerialData/purway_geotagger_app/blob/main/scripts/ci/macos_sign_and_notarize.sh).
- Secret names and account type were checked through authenticated GitHub metadata
  queries on 2026-09-14; no secret values were retrieved.
- [Apple: customizing notarization](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow).
- [Apple: packaging Mac software](https://developer.apple.com/documentation/xcode/packaging-mac-software-for-distribution).
- [GitHub: Actions secrets API](https://docs.github.com/en/rest/actions/secrets).
- [GitHub: using secrets](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

Repository links to `main` describe the inspected baseline and may change; recheck
them when resuming. This plan does not assert that existing Apple credentials are
valid until a live authentication/submission succeeds.
