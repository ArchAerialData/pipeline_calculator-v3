# Pipeline placemark IDs

The Pipelines table and the Pipeline Length Analysis spreadsheet now show
**Placemark ID** in their first column. The value comes directly from the source
`<Placemark id="…">` attribute. Missing, empty or whitespace-only attributes show
`N/A`. No OBJECTID, geometry ID, document ID or generated identifier is used as a
fallback. IDs stay as text, with their prefixes and leading zeroes preserved.

The parser carries this as `placemark_id`, and analysis results expose it as
`Placemark_ID`. Existing internal numeric indices still identify pipelines during
overlap calculations. Source IDs are retained even if a file repeats one. Sorting
the ID column compares text; sorting mileage still uses full-precision numbers.

Verified on Windows, 2026-09-14:

- All **111 WWM pipeline IDs** match the raw KMZ attributes through parsing,
  calculation, the native table and an XLSX save/reload. All 111 are distinct and
  populated. Original mileage remains **2,344.562591432312**.
- Regression cases cover plain and namespaced KML inside KML/KMZ files, multipart
  pipelines, missing/blank IDs, unrelated attribute namespaces, duplicate source
  IDs, conflicting OBJECTID and geometry IDs, sorting and spreadsheet text safety.
- The frozen EXE checks the new table heading, displayed IDs and exported IDs;
  its WWM IDs are compared against the independent source inspection before local
  replacement. Native GUI checks run on a private, undisplayed Windows desktop.

Evidence: `.validation-output/placemark-id/verification.json`,
`.validation-output/placemark-id/wwm-pipeline-ids.png`,
`.validation-output/placemark-id/wwm-pipeline-ids.xlsx`,
`.validation-output/placemark-id-full-tests.log`, and
`.validation-output/placemark-id-replacement.json`.

Source: [parser](../../src/pipeline_calculator/parsers/kml_kmz.py),
[analysis](../../src/pipeline_calculator/core/analyzer.py),
[table](../../src/pipeline_calculator/gui/tabs/pipelines_tab.py),
[export](../../src/pipeline_calculator/export/xlsx.py).
