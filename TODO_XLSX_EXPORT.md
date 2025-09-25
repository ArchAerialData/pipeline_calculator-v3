# XLSX Export — Detailed TODO

Goal: Replace the current CSV exports with a single Excel workbook (`.xlsx`) containing two sheets — `Pipeline Length Analysis` and `Pipeline Overlap Analysis` — that match the example file at `.claude/Example-XLSX/EXAMPLE_DESIRED-XLSX-EXPORT.xlsx` in structure, labels, and styling. Totals must appear at the top of each sheet. The former CSVs should not be produced when `.xlsx` is chosen.

## 1) Dependencies and plumbing
- Add runtime dependency: `openpyxl` for writing `.xlsx`.
- In `export_results()`:
  - Change dialog defaults to `.xlsx` (default extension and file type filter).
  - If `openpyxl` is missing, show a friendly prompt suggesting `pip install openpyxl` and abort export.
  - Export a single workbook with both sheets (no separate CSV files).

## 2) Sheet: Pipeline Length Analysis

Desired columns (from example):
- `Object ID (if available)` ← map from pipeline dict `OBJECTID` (use `"N/A"` if blank).
- `Polyline Name (if available)` ← map from `Name`.
- `Pipeline Lengths (US Survey)` ← map from `pipelinelength` (miles).
- `TOTAL MILEAGE` header present in column D; value placed in row 2 as a formula that sums column C.

Data mapping from current code:
- Current pipeline records (see `calculate_pipeline_lengths`): `OBJECTID`, `Name`, `Shape_Length` (meters), `pipelinelength` (miles).
- Do NOT include `Shape_Length` in the sheet (hidden/removed per request).

Totals at the top:
- Cell `D2` contains `=SUM(C2:C100000)` (same pattern as example), calculated by Excel when the file is opened.

Styling to match example:
- Header font: `Aptos Display`, size 11, bold, centered vertically; text centered horizontally except the Name column (left).
- Body font: `Aptos Narrow`, size 11.
- Header fills:
  - `Pipeline Lengths (US Survey)` column header fill: yellow `#FFFF00` (openpyxl RGB `FFFFFF00`).
  - `TOTAL MILEAGE` column header fill: green `#00B050` (openpyxl RGB `FF00B050`).
- Column widths (approximate example):
  - A: 25.11, B: 44.89, C: 29.55, D: 27.78
- Number formats / alignment:
  - Column C numeric format `0.000` (miles to 3 decimals), center aligned.
  - ID centered, Name left.
- Optional niceties: freeze header row (`freeze_panes = 'A2'`) so headers stay visible.

## 3) Sheet: Pipeline Overlap Analysis

Desired columns (from example order):
1. `Pipeline 1` ← `section['pipeline_1']`
2. `Pipeline 2` ← `section['pipeline_2']`
3. `Bundled Length (mi)` ← `section['bundled_length_miles']`
4. `TOTAL MILEAGE REMOVED` ← header; place total in row 2 via formula that sums column C.
5. `Bundled Length (m)` ← `section['bundled_length_meters']`
6. `Average Separation` ← `section['average_separation']` (meters)
7. `Segment Count` ← `section['segment_count']`
8. `Center (Long)` ← `section['center_lon']`
9. `Center (Lat)` ← `section['center_lat']`
10. `bbox` ← `section['bbox']` serialized (string)
11. `oriented_polygon` ← `section['oriented_polygon']` serialized (string)
12. `oriented_width_m` ← `section['oriented_width_m']`
13. `corridor_polygon` ← `section['corridor_polygon']` serialized (string)

Totals at the top (decision):
- Do NOT sum column C. The column `Bundled Length (mi)` lists per‑section corridor miles and will overcount in cases where more than two pipelines share a corridor or sections overlap in complex ways. The application already computes the correct survey savings that avoids double counting via its clustering logic.
- Therefore, write the app‑computed value directly in `D2`:
  - `D2 = self.current_results['overlap_analysis']['savings_miles']` (rounded to 3 decimals; number format `0.000`).
  - Keep the header text `TOTAL MILEAGE REMOVED` and the green fill to match the example.
  - Rationale: matches the “Survey Savings” figure displayed in the UI (e.g., 190.820 miles in the screenshot) and prevents misleading totals that are >1x the true savings.
  - Optional (nice‑to‑have): also write `effective_total_miles` and `savings_percentage` into adjacent cells (e.g., E1/E2, F1/F2) if we want parity with the Summary panel; default scope keeps only D2 as requested.

Styling to match example:
- Header font: `Aptos Display`, size 11, bold.
- Body font: `Aptos Narrow`, size 11.
- Header fills:
  - `Bundled Length (mi)` header fill: yellow `FFFFFF00`.
  - `TOTAL MILEAGE REMOVED` header fill: green `FF00B050`.
- Column widths (approximate example capture):
  - [44.89, 13.00, 20.33, 28.11, 21.00, 19.11, 20.78, 17.11, 20.00, 107.89, 194.55, 16.44, 255.78]
  - Note: columns with large text (`bbox`, `oriented_polygon`, `corridor_polygon`) are intentionally wide.
- Number formats / alignment:
  - `Bundled Length (mi)`: `0.000`, centered.
  - `Bundled Length (m)`: `0` (integer), centered.
  - `Average Separation`: `0.0`, centered.
  - `Segment Count`: integer, centered.
  - `Center (Long)` / `Center (Lat)`: `0.0000000` (7 decimals), centered.
  - Text columns left-aligned.
  - Optional: freeze header row (`freeze_panes = 'A2'`).

Serialization of complex fields:
- `bbox`: store as a compact string (e.g., `{min_lon: ..., max_lon: ..., min_lat: ..., max_lat: ...}`) matching the example representation.
- `oriented_polygon` and `corridor_polygon`: store as stringified lists of `(lon, lat)` pairs; no wrapping beyond the raw string.

## 4) Implementation steps in code
1. Update `export_results()` to branch on `.xlsx` and construct workbook via `openpyxl.Workbook()`.
2. Build `Pipeline Length Analysis` sheet:
   - Write headers and apply styles/fills/widths.
   - Write data rows mapped from `self.current_results['pipelines']`.
   - Write formula `=SUM(C2:C100000)` to `D2` and bold it.
3. Build `Pipeline Overlap Analysis` sheet (if bundled sections exist):
   - Write headers and apply styles/fills/widths.
   - Write data rows mapped from `self.current_results['overlap_analysis']['bundled_sections']`.
   - Serialize complex fields as described.
   - Write formula `=SUM(C2:C100000)` to `D2` and bold it.
4. Save workbook to the chosen path; show a single completion message pointing to the `.xlsx` file.
5. Keep JSON export option working if the user selects `.json` in the dialog.
6. Remove the legacy CSV branch when exporting `.xlsx` (no `_overlaps.csv`/`_summary.txt` siblings).

## 5) Validation checklist
- Sheet names exactly: `Pipeline Length Analysis`, `Pipeline Overlap Analysis`.
- Column headers match the example strings 1:1 (including capitalization and parentheses).
- Totals appear at the very top (row 2) and calculate in Excel.
- `Shape_Length` is not present on the Pipeline sheet.
- Column widths and fills resemble the example.
- Number formats and alignments are applied as specified.
- Large text columns remain readable (no truncation errors on save).

## 6) Documentation
- Update `README.md` Export section: describe the new single-workbook `.xlsx` export, the two sheets, and where totals appear.
- Note the dependency on `openpyxl` and how to install it if needed.

---

Notes captured from the provided example workbook:
- Pipeline Length Analysis headers: `Object ID (if available)`, `Polyline Name (if available)`, `Pipeline Lengths (US Survey)`, `TOTAL MILEAGE`.
- Example placed the total formula in `D2` while `C` holds per‑row miles.
- Overlap Analysis headers (13 columns) exactly as listed above; yellow fill on `Bundled Length (mi)` and green fill on `TOTAL MILEAGE REMOVED`.
