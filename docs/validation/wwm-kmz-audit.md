# WWM KMZ calculation and corridor audit

September 14, 2026. **Confirmed corridor defects fixed; numeric outcomes unchanged.**

Input: `C:\Users\rbake\Downloads\Q3 - WWM Pipelines.kmz`. SHA-256: `b927593def8f50ff08028f00fd141ca6f0d7483c2331f9df9731d16735a3f937`. Source Python fingerprint: `d217688091edcd53f8b81ab11208f6e1844afe616fe4151eff89c0ae23f07c4b`.

Tested the actual file through the shared GUI-to-backend controller at 14 settings. All 13 supported runs passed independent overlap-search, qualification, savings, and exported-geometry checks. The 2 m full-file run deliberately retained its existing explicit workload-limit result. No input geometry was modified.

## Verified default results

| Measure | Value |
| --- | ---: |
| Original mileage | 2,344.562591 US survey miles |
| Mileage removed | 99.353948 US survey miles |
| Adjusted mileage | 2,245.208643 US survey miles |
| Pairwise bundled mileage | 100.693000 US survey miles |
| Qualified pairwise sections | 95 |

Pairwise bundled mileage is not mileage removed: the savings calculation prevents counting a pipeline segment repeatedly and requires mutually compatible groups. Its nearest-first grouping is a conservative estimate, not an optimal route calculation.

## What was checked

- **KMZ structure:** one `doc.kml`, 111 pipeline records/paths, 172,247 coordinate tuples. All original coordinate lists match the parser output exactly. No dropped path, linked document, invalid-coordinate notice, or duplicate name was found. Independently summing original GRS80 geodesic edges gives 3,773,215.285575587 m, agreeing with the app within a nanometre of floating-point summation. This is agreement on the calculation, not input survey accuracy.
- **Configuration:** actual native input variables, clamping/correction, shared parameter parser, controller constructor arguments, and result settings agree. Defaults remain 15 m range, 200 m minimum continuous length, 5 m sampling, and 15 degrees.
- **Sample search and qualification:** a separate per-pipeline spatial search, vectorized geodesic predicates, and SciPy graph connectivity agree with all accepted/rejected sample pairs and qualified sections across every supported matrix row. Default: 75,317 matching sample pairs, no missing/extra pairs, 95 sections.
- **Savings:** an independently implemented explicit-partition calculation reproduces the documented nearest-first, mutually compatible grouping policy for every supported run. Default savings: 159,895 m. Original minus savings equals adjusted mileage; segment and pairwise totals agree; the same result values feed GUI and XLSX exports.
- **Corridor KMLs:** all **1,469 exported files** parse and have valid, closed, non-self-crossing polygons. Shapely independently checks continuous portions of the original source paths, including vertices, endpoints and intermediate samples, against the serialized 7-decimal KML outlines. After fixes: **zero clipped corridors and 0 m uncovered path**, using a 2 cm tolerance for serialization/projection noise. These are geometric checks; no Google Earth viewer was opened.

## Confirmed fixes

1. **Endpoint clipping:** representative midpoint pairs could omit a partner pipeline's final matched segment. Default exports clipped 50 of 95 outlines in the original sample checks, with up to 2.398 m outside; 50 m sampling reached 21.470 m. Outlines now retain original section endpoints and bend vertices, extend their caps to the true qualified extent, and fall back to an enclosing rectangle if a curve cuts through a source path.
2. **False zigzag detection:** the old heuristic mistook regular coarse sampling for zigzags. At 20 m, 79 of 87 outlines became broad rectangles; now all 87 use valid curves with full qualified-path coverage. Validity/work-budget fallbacks remain and are disclosed in KML descriptions (two at defaults, 45 across the matrix).
3. **Import order:** a fresh parser-first import exposed a circular import through the eager core analyzer export. The public analyzer export now resolves lazily.
4. **Rapid close:** native settings validation exposed double deletion of child-owned Tk timer commands when closing immediately. Root shutdown now cancels timer execution while letting each owning widget remove its command.

![Before/after corridor evidence](../../.validation-output/wwm-audit-final/corridor-before-after.png)

The top view shows a default endpoint; the bottom shows the longest 20 m corridor. Blue/orange are the original qualified source paths. Rectangles can include substantial extra area even when their mileage label is correct.

## Settings matrix

All distances below are metres except the mileage columns; angle is degrees. Each row has zero invalid or clipped exported corridors.

| Case | Range | Minimum | Step | Angle | Removed miles | Adjusted miles | Sections | Rectangle fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 15 | 200 | 5 | 15 | 99.353948 | 2245.208643 | 95 | 2 |
| range10 | 10 | 200 | 5 | 15 | 18.811975 | 2325.750616 | 31 | 1 |
| range25 | 25 | 200 | 5 | 15 | 511.962236 | 1832.600356 | 291 | 6 |
| minimum100 | 15 | 100 | 5 | 15 | 106.651938 | 2237.910653 | 183 | 2 |
| minimum500 | 15 | 500 | 5 | 15 | 89.937086 | 2254.625505 | 42 | 1 |
| angle5 | 15 | 200 | 5 | 5 | 96.417975 | 2248.144616 | 123 | 0 |
| angle30 | 15 | 200 | 5 | 30 | 100.121340 | 2244.441251 | 90 | 6 |
| step4 | 15 | 200 | 4 | 15 | 99.260121 | 2245.302470 | 99 | 6 |
| step10 | 15 | 200 | 10 | 15 | 100.307751 | 2244.254840 | 91 | 1 |
| step20 | 15 | 200 | 20 | 15 | 100.425811 | 2244.136780 | 87 | 0 |
| step50 | 15 | 200 | 50 | 15 | 100.537658 | 2244.024934 | 98 | 10 |
| strict | 8 | 500 | 10 | 5 | 3.840066 | 2340.722525 | 7 | 0 |
| broad | 30 | 100 | 10 | 30 | 526.747734 | 1817.814858 | 232 | 10 |
| step2-limit | 15 | 200 | 2 | 15 | Unavailable: segment limit | Unavailable | — | — |

## Accuracy limits and investigated apparent gaps

The displayed numbers are internally correct for the sampled model; they are not exact continuous-boundary measurements. Changing only the step from 5 m to 4 m changes removed mileage by -0.093827 miles (about -0.0944% of the default savings). At 20 m it changes by +1.071863 miles. Coarse sampling can smooth short bends or bridge a short nonqualifying gap; a minimum-length threshold can amplify that small boundary change into a whole section appearing or disappearing.

An additional original-edge comparison gave about 100.710 pairwise miles at defaults versus 100.693 sampled miles. That comparison has different joint/continuity handling and is not an exact savings oracle. Its apparent omissions were investigated on the original pipeline pairs at 1, 4, 5 and 10 m spacing:

- **BRP Agua Dulce LP Header / BXP ML-6 Freer to Agua Dulce:** the apparent 303 m area consists of two runs separated by a nonqualifying gap. At 1 m they measure 149 m and 150 m; at defaults, 145 m and 155 m. Each is below the 200 m minimum, so neither should produce a default corridor. A 10 m step bridges the gap and reports 300 m, demonstrating the coarser estimate's limitation.
- **MXP Stanton Lateral / Stanton Loop Lateral:** an apparent 116 m area similarly separates into 74/21/20 m runs at 1 m spacing and 70/40 m at 5 m. It does not meet a 100 m continuous minimum at the finer settings; 10 m sampling joins it into 120 m.

Short trailing remainders below the selected step remain in original/adjusted mileage but are not eligible for sampled overlap savings. The original KMZ structure is not responsible for these threshold effects. Full-file 2 m sampling requires more than the supported 1,000,000 segments, so the app reports unavailable overlap values rather than a misleading successful zero.

**No unexplained missing sample matches or qualified-path clipping remain in these checks.** Exact real-world overlap boundaries and Google Earth rendering were not certified.

## Validation and evidence

274 tests passed, including 14 added regression cases for endpoint phase/reversal, coarse curves, containment across concavities, bounded checks, original-path extents, parser-first imports, and actual GUI settings/close behavior. All native tests ran on undisplayed Windows desktops.

- [Machine-readable verification summary](../../.validation-output/wwm-audit-final/verification-summary.json)
- [Full settings matrix](../../.validation-output/wwm-audit-final/matrix.json)
- [Default corridor checks](../../.validation-output/wwm-audit-final/default/corridor-checks.json)
- [Default output workbook](../../.validation-output/wwm-audit-final/default/results.xlsx)
- [Boundary investigation](../../.validation-output/wwm-boundary-investigation.json)
- [Full test log](../../.validation-output/wwm-audit-full-tests.log)
- [Packaged WWM smoke](../../.validation-output/wwm-audit-packaged-wwm.json)
- [Startup check](../../.validation-output/wwm-audit-startup.json)
- [Installed EXE hash and previous-build backup](../../.validation-output/wwm-audit-replacement.json)

The replacement Windows EXE passed the packaged WWM calculation and live-results
transition, plus the normal startup check. `dist/Pipeline_Calculator_v4.exe` now
contains that tested build; the previous executable is preserved in `dist/archive`.
SHA-256: `F7CD1FC45504BD4BFE532CFCDDF2701844D9163295F0F214DE0ED3D59CA651BF`.

Implementation: [source extents/containment](../../src/pipeline_calculator/core/corridor_coverage.py), [outline construction](../../src/pipeline_calculator/core/overlap.py), [regressions](../../tests/test_corridor_coverage.py).

Run the opt-in audit from the repository root (Shapely is a validation-only dependency and is not added to the app):

```powershell
.venv/Scripts/python.exe -m pip install --target .validation-output/wwm-audit-deps --no-deps shapely==2.1.2
.venv/Scripts/python.exe scripts/validation/audit_dataset.py "C:\Users\rbake\Downloads\Q3 - WWM Pipelines.kmz" --geometry-deps .validation-output/wwm-audit-deps --output .validation-output/wwm-audit-final --reference
```
