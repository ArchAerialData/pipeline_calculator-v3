# Corridor construction capacity

September 18, 2026. The cumulative corridor construction allowance is now
**10,000,000 work units**, increased from 2,000,000. Per-operation, per-section,
output, chart and boundary-query limits remain unchanged, as do padding,
accuracy, containment and numerical mileage calculations.

The counter includes repeated projections, geometry construction, spatial-index
queries and validation. It is neither a count of distinct input/output vertices
nor a measurement of RAM. One counter spans Combined and all states. A failed
map keeps its numerical results and an explicit omission diagnostic.

## Measured input

The supplied WWM Q3 input was analyzed from an isolated snapshot of commit
`b1a456a681d4e367ec7e54028492e91d16b911dd`, changing only the cumulative work
allowance in the probe. Its SHA256 is
`b927593def8f50ff08028f00fd141ca6f0d7483c2331f9df9731d16735a3f937`.
The private input and full analysis output are not copied into the test suite.

| Measurement | Original 2M allowance | 10M confirmation |
| --- | ---: | ---: |
| Combined maps available | 50 of 95 | 95 of 95 |
| Texas maps available | 0 of 95 | 95 of 95 |
| New Mexico qualifying maps | 0 | 0 |
| Charged work / peak preflight | Exhausted | 7,263,267 / 7,263,267 |
| Retained polygon vertices | 20,613 | 70,366 |
| Construction charts | — | 380 |
| Boundary queries | — | 471 |

A preliminary 20M run completed with identical work/output counts. The 10M
confirmation therefore leaves approximately 38% headroom above this input's
measured requirement. This is evidence for this input and parameter set, not a
guarantee that all larger or more complex inputs fit.

The 10M run took 89.33 seconds for full analysis, including 15.06 seconds in
instrumented map generation. Its peak working set was 790,786,048 bytes
(approximately 754 MiB); peak process commitment was 2,318,938,112 bytes
(approximately 2.16 GiB). These measurements include the entire analysis, not
just maps. They do not establish a direct relationship between work allowance
and memory consumption or a cross-platform performance guarantee.

Original mileage, adjusted mileage, savings and numerical section records were
unchanged. The independent review of the 20M output additionally compared the
complete source fragment ledger/reconciliation and checked 157 Texas polygons,
including eight holes, against the original state boundary resource: every
polygon was valid and contained; no warnings or errors remained. The 10M
confirmation reports the same complete map counts and unchanged numerical
results.

Local receipts remain under `.validation-output/wwm-corridor-review/`:
`summary.json`, `budget-20000000/summary.json`,
`budget-10000000/summary.json`, and their analysis/measurement files. They are
local evidence rather than a CI fixture or published platform certification.

After implementing the default allowance, a fresh run against the current
application source (without overriding any budget) completed in 87.67 seconds.
All 95 Combined maps and 95 Texas maps were ready. Original/adjusted mileage,
savings, polygon coordinates and reconciliation matched the isolated 10M
confirmation exactly. Receipt:
`.validation-output/corridor-header-fix/wwm-current-summary.json`.

## Fast regressions

[Capacity tests](../../tests/test_corridor_budget_capacity.py) simulate already
completed work by advancing the accounting counter, then run the actual
Combined and state geometry builders above the former cap. A second case
exhausts the new allowance during later map work and verifies that previously
generated geometry, numerical results and retained-output accounting survive.
No large private input or slow full-network analysis is needed in CI.

Existing geometry tests continue to enforce the independent per-operation,
per-section, retained-output and containment guards. Source verification is
separate from packaged Windows/macOS validation.

## Desktop presentation and local Windows verification

Unavailable rows explain the recorded failure using structured diagnostic codes,
including resource limits preserved through state clipping. The map action stays
disabled. Selecting the row shows a concise explanation below the table; short
windows use a keyboard-accessible **Map details** action in the existing pager.
Neither presentation claims an incomplete numerical analysis has completed.

The Combined/state selector and helper text are centered together within the
existing 50-pixel navigation row, with measured native control height and tighter
vertical padding. Compact navigation retains its existing 34-pixel row. Header
checks cover resizing and scaling through 250%, without clipped control text or
extra header height.

Verification: 121 focused core/geometry tests, 46 corridor launch/consumer/UI
tests, and 36 related header/navigation/settings UI tests passed. Full-page
omission tests include Combined and Texas at 640×360 and 100%/150% scaling;
native captures were also inspected.

The local Windows EXE was rebuilt and both packaged GUI entrypoints passed
offline smoke checks. Build input hashes were unchanged through verification.
The verified binary was copied to `dist/Pipeline_Calculator_v5.exe`; provenance,
SHA-256, native captures and smoke reports are retained under
`.validation-output/corridor-header-fix/`. This does not assert a new macOS build.
