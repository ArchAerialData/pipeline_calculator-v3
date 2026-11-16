# Customer shapefile ZIP inspection fixtures

These are exact copies of the two ZIPs supplied for GIS import planning on
November 16, 2026. The user authorized repository fixture copies. The originals
remain unchanged in their download locations. Import tests stage captured bytes
in temporary directories; users do not need to extract either archive.
These contain customer data, not synthetic public examples. No new publication
or redistribution was performed as part of this work.

| Fixture | Bytes | Original and copy SHA-256 |
| --- | ---: | --- |
| `RF_Lehman_Pipes.zip` | 91,284 | `10b57d3b76c2e23304233fa559d9971c2956052e5e55a56ee327b11cd7fee1e4` |
| `Verdun_Pipes.zip` | 508,361 | `3dfea6159b5e7ef9bec27118dde484d3ac2c04f9b460384304c01a25359af44d` |

Both archives contain one complete `.shp/.shx/.dbf/.prj/.cpg` dataset, plus
optional `.sbn/.sbx` spatial indexes and `.shp.xml` metadata. ZIP CRC checks pass.
The metadata XML is treated as opaque bytes: it is not parsed or treated as
instructions. The inspected `.prj` declares geographic NAD83 (`EPSG:4269`), and
the `.cpg` declares UTF-8. Each SHP header declares PolyLineZ (13).

| Observed property | RF Lehman | Verdun |
| --- | ---: | ---: |
| Physical SHP / SHX / DBF records | 228 | 1,534 |
| Active DBF rows | 228 | 1,534 |
| Non-null geometry records | 228 | 1,530 |
| Null geometry records | 0 | 4 |
| Multipart geometry records | 6 | 32 |
| Separate line parts | 234 | 1,584 |
| XY vertices | 3,458 | 31,277 |
| Independent horizontal geodesic reference, meters | 225,341.9297198959 | 1,770,249.6445552604 |

Verdun's null records have zero-based physical indices `1514, 1515, 1520, 1527`
(one-based ordinals `1515, 1516, 1521, 1528`). These are positional SHP/DBF
identities, not customer identifier attributes. All four retain active DBF rows.
The inspector reads every geometry and every DBF row separately so null or
deleted rows cannot be hidden by a paired iterator.

RF's Z ordinates are all zero and M ordinates are missing. Verdun has 842 nonzero
Z ordinates and 835 finite M ordinates. Their meaning and units have not been
established. Horizontal reference measurement uses XY only, with no joining
between separate parts. `GlobalID` is nonempty and unique in both datasets;
Verdun `PLINE_ID` has four empty entries and duplicate values. Identifier
summaries in the report contain counts, not raw attribute values.

## Reproduce the evidence

From the repository root, using the development environment:

```powershell
.venv\Scripts\python.exe scripts/validation/inspect_gis_archives.py
```

This reads these fixture ZIPs in memory and writes
[`docs/validation/gis-import-samples.json`](../../../../docs/validation/gis-import-samples.json).
The hash-pinned [inspector](../../../../scripts/validation/inspect_gis_archives.py)
refuses changed or unknown samples. It records archive inventory, CRC status,
complete record counts, null indices, parts, vertices, bounds, Z/M summaries,
field schema, identifier uniqueness, source CRS, dependency versions, and an XY
digest. The report specifies the digest's binary record/part delimiters precisely.

To additionally verify the untouched supplied originals before and after:

```powershell
.venv\Scripts\python.exe scripts/validation/inspect_gis_archives.py --verify-originals 'C:\Users\rbake\Downloads\Noah\input\RF_Lehman_Pipes.zip' 'C:\Users\rbake\Downloads\Noah\input2\Verdun_Pipes.zip'
```

The recorded inspection report includes that successful original/copy comparison.
The ordinary command omits that optional section and does not require access to
the original downloads. Within an unchanged dependency environment, identical
command arguments produce a deterministic JSON report.

## Reference limits

The independent reference uses each actual SHP part with
`pyproj.Geod(ellps="GRS80").line_length` and sums using `math.fsum`. It explicitly
pins NAD83-to-WGS84 operation `EPSG:1188` (represented by `+proj=noop`) only after
checking the entire measured geographic extent is inside that operation's area
of use. It verifies every XY remains identical. The operation's stated **4 m**
accuracy is not source survey accuracy and is not a mileage error bound. This is
a sample-specific reference, not a global CRS transformation policy.

This evidence is independent of the application parser/analyzer; it uses the
same geodesic library family as the app and is not a second geodesic algorithm.
The archives demonstrate real multipart PolyLineZ and null geometry cases. They
do not establish application import compatibility, GPKG support, projected CRS
handling, points, polygons, mixed-layer selection, overlap/state results, or
packaged execution. No automated application regression tests were added during
this planning step. The future test suite should combine these preserved
archives with small, independently authored synthetic fixtures for those gaps.
