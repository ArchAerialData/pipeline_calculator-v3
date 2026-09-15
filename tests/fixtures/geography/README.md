# Geography regression fixtures

## Adamas NG pipeline ROW

`adamas_ng_pipeline_row.kmz` is a byte-for-byte copy of the user-provided
`Adamas - NG_PIPELINE_ROW.kmz`. The source file remains in Downloads. The
[expectation manifest](adamas_ng_pipeline_row.expected.json) records its SHA-256,
archive size, XML geometry counts and independently inspected behavior.

Despite the filename and `Pipeline-*` feature names, the KMZ contains **281
polygons, no LineStrings and no points**. All 281 polygons are valid and lie wholly
within one supported state: Texas 57, Louisiana 197 and Wyoming 27. This state
classification describes polygon coverage; it is not pipeline-mileage analysis.

Use this archive to guard against accidentally counting polygon outlines as
pipeline centerlines. The existing parser returns zero pipelines and zero mileage,
with `unsupported_geometry` warnings and a `no_supported_features` error. Analysis
must be marked incomplete, and the UI should explain the unsupported input clearly.
Do not convert these rings to LineStrings or save current perimeter lengths as
expected pipeline mileage.

The companion centerline fixture below provides the intended positive test.

## Adamas NG centerlines

[adamas_ng_pipeline_row_centerlines.kmz](adamas_ng_pipeline_row_centerlines.kmz)
is a byte-for-byte copy of the subsequently supplied
`Adamas - NG_PIPELINE_ROW-Centerlines.kmz`; the source remains in Downloads.
Its [expectation manifest](adamas_ng_pipeline_row_centerlines.expected.json) records
the checksum, independently computed source lengths and per-feature state assignments.

Direct XML inspection found **281 features, 4,827 LineString paths, 48,327 vertices,
67 multipart features and 15 closed paths**, with no polygon geometry or invalid
coordinates. All 281 XML IDs repeat `ID_00000`; source identities must remain distinct.
Multipart paths must not be joined across gaps.

| State | Source pipelines | Paths | Original meters |
| --- | ---: | ---: | ---: |
| Louisiana | 197 | 3,563 | 1,255,195.5030877413 |
| Texas | 57 | 452 | 335,084.0137085223 |
| Wyoming | 27 | 812 | 375,638.3464031983 |
| Total | 281 | 4,827 | 1,965,917.863199462 |

Every complete source path belongs to one state; no source feature spans states.
The application reports zero crossings, shared mileage, outside mileage and
unresolved mileage. Full analysis and Combined/state KMZ mileage round trips passed
against the fixed audit baseline. Original total is approximately **1,221.562 US
survey miles**. State overlap savings were zero at default settings; this is an
observed result, not an independently established overlap oracle.

Independent expectations use direct XML parsing, GRS80 geodesic edge lengths and
whole-path state containment after geodesic densification to at most 100 m spacing.
That containment check supports this fixture's state assignments; it does not certify
sub-centimeter boundary accuracy or the upstream method used to create centerlines.

Add regressions for parsing, unique internal identities despite repeated XML IDs,
disconnected multipart paths, state source counts, zero crossings, per-source and
total conservation, and export/reimport mileage. Retain synthetic cross-border and
shared-border cases because this fixture does not exercise those workflows.

No production code or automated test files were changed during this audit because
another task was editing the shared workspace. Implement the proposed tests with the
[resolution plan](../../../STATE_BOUNDARY_ANALYSIS_RESOLUTION_PLAN.md) after that work
has finished and its final state has been reconciled.
