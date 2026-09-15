# Geographic comparison findings

Baseline: `71da499d5756648ae395660f0a241ea00edbea4f`. Boundary archive:
`55d4bb65ae174f1b1cf9fd4e012ef6b8a166ae8ee4538a0a1205ce2d45697539`.
Machine-readable measurements and archive hashes are in [findings.json](findings.json).
No application code or main-fixture expected values are changed by this reproduction.

## Confirmed application defect: longitude normalization moves a native vertex

[02_native_vertex_shift.kmz](02_native_vertex_shift.kmz) is the **corrected**
`02_endpoint_touch`: one 363.25 m geodesic terminating at the exact native
boundary vertex `(-103.041703, 36.478411)`. Direct inspection of the supplied WKB
proves that both New Mexico and Texas contain that exact endpoint. All positive
source mileage is New Mexico, and the expected crossing count is zero.

The application's boundary normalization changes the vertex longitude in **both**
polygons from `-103.041703` to `-103.04170300000001`. Its radian/degree round trip
in `_unwrap_ring` changes already canonical continental coordinates by one
floating-point unit. Consequently its New Mexico polygon no longer contains
the original native endpoint, and the application assigns a
`1.2735767995764036e-9 m` tail to Texas, reports an extra represented source in
Texas, and counts a crossing. The reversed two-vertex source also falsely
counts one crossing. Main fixture 02 therefore reports **10 application
crossings versus 9 independently established crossings**, even after the draft
construction correction described below.

[findings.json](findings.json) retains the native and application vertices,
membership checks, and full source interval ledgers. The respective NM/TX
native-versus-application symmetric-difference areas are
`1.4180708079720763e-14` and `4.0048059439683016e-14 deg²`. These tiny differences
are not a surveyed-accuracy claim; they demonstrate that the application's
numerical model does not preserve the supplied boundary coordinates. See
[boundaries.py](../../../../../../src/pipeline_calculator/core/geography/boundaries.py)
at lines 27–32. The reference and fixture remain unchanged following this
application finding.

## Confirmed application defect: exact endpoint creates unresolved mileage

[04_endpoint_roundoff.kmz](04_endpoint_roundoff.kmz) contains one source, one path,
and two vertices. It is the original serialized 4 m approach edge of
`04_shared_qualification_route`, vertices 1 and 2, copied without rounding.

The ending longitude is exactly the canonical shared meridian `-103.064732`.
The preceding positive-length geodesic is exclusively Texas; the endpoint also
touches New Mexico. Independent original length is `4.000000000333733 m`.
Expected unresolved mileage and crossings are both zero.

The application creates an additional unresolved interval of
`8.881784197001252e-16 m` and reports geography `incomplete`. Its two final
stations are `4.000000000333732` and `4.000000000333733`. The generated cut
coordinate has latitude `32.749625281317385`, while the exact source endpoint is
`32.7496252813174`. This is a numerical duplicate of an already known endpoint,
not a genuine short state visit. Reversing the same two serialized vertices
produces `complete` geography and exactly zero unresolved mileage; see
[04_endpoint_roundoff_reversed.kmz](04_endpoint_roundoff_reversed.kmz).

The cause is consistent with the inspected implementation: the partitioner
accepts every computed root strictly between `0` and the edge length, then
deduplicates only exactly equal floating-point stations. The resulting interval
has multiple state memberships without proven positive shared coincidence and
becomes unresolved. Any positive unresolved mileage makes state analysis
incomplete. See [partition.py](../../../../../../src/pipeline_calculator/core/geography/partition.py)
at lines 217–240 and [state_analysis.py](../../../../../../src/pipeline_calculator/core/state_analysis.py)
at lines 240–247. The appropriate repair should preserve canonical endpoint
identity; a blanket short-interval filter could erase the real 10 cm crossing
retained by the main fixture and is not established as correct by this evidence.

## Rejected construction and independent-reference correction

An earlier draft of `02_endpoint_touch` ended at an interpolated point on a
nonmeridian boundary. After serialization, its exact binary coordinate was
slightly inside Texas, not exactly on the boundary. An 80-digit calculation of
the determinant against the native edge gave
`3.104315856863637227743422911823735488301423401935608126223087310791015E-17 deg²`;
direct polygon membership was Texas only. The initial application's extra
crossing was therefore **not established as an application defect**.

This investigation also found an independent-reference defect: merging a root
bracket that overlapped the source endpoint could retain the earlier station,
trimming a submicrometer tail that was smaller than the conservation threshold.
That was not a valid zero-length touch. The reference now preserves canonical
endpoints, evaluates endpoint determinants with 80-digit exact-binary inputs,
requires multiple verified memberships for touches, and fails closed when root
uncertainty overlaps an endpoint without proof of exact boundary identity.
The rejected numerical construction is an executable negative control in
[reproduce.py](reproduce.py), not a regression golden.

The generator now uses the exact native boundary vertex
`(-103.041703, 36.478411)` for this touch. Expectations were regenerated from the
changed serialized archive because the independent proof invalidated the draft
construction, not to match an application result. Every other main archive kept
the same bytes during this correction.

## Reproduce

From the repository root with the suite requirements installed:

```powershell
python tests/fixtures/geography/pipeline_kmz_regression_suite/validation/reproductions/reproduce.py
```

The script deterministically writes all four small archives, establishes their
independent references before importing application calculations, checks that
the rejected interpolated endpoint fails closed, and updates `findings.json`.
