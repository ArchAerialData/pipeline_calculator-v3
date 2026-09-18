# August 2023 KMZ: stored mileage versus supplied geometry

Verified September 17, 2026 against `Pipelines August 2023.kmz`, SHA-256
`92bea274a0e588e2611e4a46b96b8a8f48d3b26f9089533a1b9dfbbd89f1eb4c`.

The repair includes every line present in this file. Its `MILES` attributes sum to
648.601713, but its supplied line coordinates measure 572.1250571784351 US survey
miles. The 76.476656-mile discrepancy is concentrated in four source records;
it is not explained by ordinary decimal precision differences.

## Record comparison

| Source record | Stored MILES | Calculated miles | Stored minus calculated |
| --- | ---: | ---: | ---: |
| 31_MARKHAM TO RAMSEY, 8" C2 | 48.522005 | 41.669920 | 6.852085 |
| 32_RAMSEY TO DEER PARK, 6" C3 | 54.797756 | 3.811259 | 50.986497 |
| 50_SR - HWY 60, 10" C2 | 43.037097 | 36.635168 | 6.401929 |
| 51_STRAT RIDGE TO BAY CITY (IDLE), 4" C2 | 49.081095 | 36.873119 | 12.207976 |
| Other 38 records together | 453.163760 | 453.135591 | 0.028169 |
| **All 42 records** | **648.601713** | **572.125057** | **76.476656** |

The four outliers explain 99.963% of the net discrepancy. Every other record
agrees within 0.031% of its stored mileage (maximum absolute difference
0.005658 miles). Both multipart records and every record marked idle are included.
The Ramsey-to-Deer-Park record contains just one 16-vertex path, measuring
6,133.639 meters; there is no additional path in that record to recover.

## Preservation and independent checks

- The archive contains one document, `doc.kml`, with 42 placemarks, 44 LineStrings,
  44 coordinate blocks and 2,139 vertices. It has no NetworkLinks or other
  Point/Polygon/Track geometry. All supplied altitudes are zero.
- The only repair is a 54-byte namespace declaration inserted at byte 211.
  Removing that insertion recovers the original XML byte for byte.
- Direct XML enumeration of every coordinate block matches the production
  parser's complete path sequences, by source order, vertex for vertex. Repeated
  placemark IDs do not cause deduplication.
- Direct GRS80 measurement agrees with each production pipeline's measured
  meters. A separate math-only Vincenty calculation agrees with the overall
  pyproj result within 0.000018 meters for this dataset; all edges converged.
- These are original lengths, before any overlap deduction or state allocation.
  The original customer file was unchanged.

Changing GRS80 to WGS84 or switching survey/international miles does not explain
this discrepancy. Web Mercator planar distances were also compared per record
and disagree with the stored fields. Its superficially closer aggregate total
is not evidence of the client's calculation method. Web Mercator's length
distortion is documented by
[Esri](https://developers.arcgis.com/documentation/spatial-analysis-services/geometry-analysis/length-and-area/).

## Interpretation and client follow-up

The evidence proves that the repaired analysis includes the geometry supplied in
this KMZ. It cannot prove that the client supplied their complete system. Stale
mileage attributes or an export containing only portions of the intended routes
are possible explanations, but this file does not establish which is correct.

Suggested client request:

> Please verify the four pipeline records listed above. Their MILES values do not
> match the line geometry in the supplied KMZ. In particular, the Ramsey-to-Deer-Park
> record lists 54.797756 miles but contains approximately 3.811259 miles of line
> coordinates. Please provide a fresh KML/KMZ containing the complete intended
> routes and mileage attributes recalculated from that same geometry, or confirm
> that the supplied geometry is complete and explain what the stored mileage
> fields represent.

Stored mileage is useful corroborating evidence and can identify records for
review. It must not be used to scale coordinates, invent missing segments,
override measured lengths, or declare a repair complete merely because totals
are close. Repair acceptance remains based on exact geometry preservation and
complete extraction of the geometry actually present.

## Reproduction and retained evidence

- [All 42 records as CSV](august-2023-mileage-comparison.csv).
- [Production comparison and coverage assertions](august-2023-mileage-comparison.json).
- [Sample-specific reproducible inspection](../../scripts/validation/inspect_august_2023_miles.py).
- [Independent measurement evidence](august-2023-mileage-independent.json).

Run from the repository root, supplying the original customer file:

```powershell
.venv/Scripts/python.exe scripts/validation/inspect_august_2023_miles.py 'C:\Users\rbake\Downloads\Pipelines August 2023.kmz' --report docs/validation/august-2023-mileage-comparison.json --csv docs/validation/august-2023-mileage-comparison.csv
```

This investigation changes documentation and adds a diagnostic script; it makes
no production measurement or repair changes, so no executable rebuild is needed.
