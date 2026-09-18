# Terminal sampling precision amendment

The September 2026 release audit found a cross-platform overlap difference in
`04_shared_border_and_near_border__redundant_vertices.kmz`: Windows returned
1,865 m of combined savings and the hosted macOS ARM build returned 1,870 m.
The inputs and dependency versions matched. Two nominal 430 m paths measured
429.9999999966561 m on Windows, only 3.34 nanometres below a full 5 m sample.
The previous 1 nm comparison could therefore omit a whole sample. A local
in-memory experiment changing that comparison to 10 nm reproduced the macOS
result without modifying the input.

The explicit numerical policy is now:

```
terminal allowance = min(0.000001 metre, sample length × 0.000001)
```

Only the final sample of each disconnected path can use this allowance. Its
endpoint is clamped to the real source endpoint. Intermediate residuals are
preserved, so adding vertices cannot repeatedly claim the allowance. The
relative cap prevents tiny legal sample lengths from gaining phantom samples.
Original coordinates, original mileage, clipping and state ownership are not
changed. This is numerical stability, not a claim about survey accuracy.
Ordinary shorter tails remain included in original mileage and excluded from
sampled overlap savings. Corridor extraction uses the same endpoint allowance
and retains the real source endpoint.

The independent fixture reference remains a separate whole-path cumulative
chainage implementation, with no imports from production. Its version is now
`sampled-contract-2`; the reference profile records the allowance. Exact sample
counts and the existing 1e-7 m savings comparison remain enforced.

Independent reconstruction changed only these numeric goldens:

| Fixture | Scope | Samples before → after | Savings before → after |
| --- | --- | --- | --- |
| 04, redundant-vertex variant, reversed variant | Combined | 751 → 753 | 1,865 → 1,870 m |
| Same three inputs | New Mexico | 221 → 222 | 425 → 430 m |
| Same three inputs | Texas | 303 → 304 | 425 → 430 m |

The other four core inputs and the 96-source stress input have unchanged
numeric results; their reference metadata records the amended policy. All
archives, geometry ledgers, original mileage and qualifying section counts
were verified unchanged before publishing the new goldens.

The corridor packaged smoke control also has two 1,199.9999997792493 m paths.
Their 0.221 µm deficit now qualifies for the final sample, increasing that
section from 1,195 to 1,200 m and total savings from 2,285 to 2,290 m. The
599.9999986597815 m control remains at 595 m because its deficit exceeds 1 µm.
These expectations were calculated by the independent reference before
updating the smoke assertions.

Focused verification covers both sides of the terminal boundary, genuine
short tails, redundant vertices, duplicate terminal vertices, tiny steps,
segment limits and corridor endpoint containment. The seven complete KMZ
goldens run in both analysis modes with unchanged exact assertions. Hosted
Windows/macOS validation remains a separate release gate; local agreement
alone is not proof that a packaged build has passed.

Sources: [sampling implementation](../../src/pipeline_calculator/core/segmentation.py),
[independent reference](../../tests/fixtures/geography/pipeline_kmz_regression_suite/reference/sampled.py),
[direct regressions](../../tests/test_segmentation.py),
[frozen integration checks](../../tests/test_state_kmz_regression_suite.py).
