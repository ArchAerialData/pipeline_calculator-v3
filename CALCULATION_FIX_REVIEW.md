# Calculation fixes ready for review

Historical record of the first correction pass. The subsequent
[follow-up audit](FOLLOWUP_AUDIT.md) supersedes the neighborhood-sharing method
and the dateline-rendering limitation described below; it also expands validation.

Prepared September 9, 2026. Changes are uncommitted on
`codex/pipeline-ci-signing-setup`; no commits, merges, releases, signing operations,
or branch deletions were performed. The starting files matched the merged main
revision `05a8d6c` (branch commit `e7997b3`).

## Results

All six reproduced findings have implementation changes and regression coverage.

| Reproduction | Before | After |
| --- | --- | --- |
| Two 300 m lines, 2 m apart, 8 m range | 890 m bundled | 300 m bundled, 300 m savings |
| Same lines, default 15 m range | No bundled sections | One 300 m section, 300 m savings |
| Two 100 m lines, 200 m minimum | No sections but 100 m savings | No sections, zero savings, 200 m effective |
| Parallel lines 14.99 m apart near equator, 15 m range | No matches | Qualifying overlap detected; outside-limit controls rejected |
| Internal KMZ link containing `layers/../pipes/data.kml` | Pipeline omitted | Pipeline loaded; links leaving archive root remain rejected |
| Injected overlap calculation failure | Totals returned without an error diagnostic | Incomplete status, explicit error, savings unavailable in Excel |
| Invalid interior LineString/Track vertex | Surviving points connected across the invalid vertex | Affected geometry rejected; valid sibling geometries retained; incomplete status |

The first two rows share the same many-to-many matching defect.

## Implementation decisions

- [bundling.py](src/pipeline_calculator/core/bundling.py) is the shared source for
  section qualification and savings. Matches form connected components along
  both paths' segment indices. Each covered segment counts once per section;
  missing segment indices and separate coordinate paths break continuity.
- Minimum parallel length applies to unique coverage on **both** paths. The
  shorter covered length is the reported common length. Repeated neighboring
  matches cannot increase it.
- [analyzer.py](src/pipeline_calculator/core/analyzer.py) derives effective length
  from the savings of those same qualified sections, removing the previous second
  independent search. The legacy effective-length helper delegates to the shared
  rules and now also honors minimum length.
- Savings retain the segment-neighborhood allocation model: covered length is
  shared among participating pipelines. For unequal coverage, the longer side is
  scaled to the common length. Repeated coverage from the same partner is not
  added twice. Three coincident lines retain one line's effective mileage.
- [spatial.py](src/pipeline_calculator/core/spatial.py) indexes ellipsoid-surface
  Cartesian coordinates. Their chord distance is no greater than geodesic surface
  distance; this avoids approximate-map exclusions before the final geodesic
  range check. Tests include the equator, high latitude, and a dateline crossing.
- [kml_kmz.py](src/pipeline_calculator/parsers/kml_kmz.py) normalizes internal link
  paths after joining to the referring document. Archive members are read without
  extracting them. Absolute paths and paths escaping the archive remain excluded.
- Invalid coordinates reject the entire affected LineString or gx:Track rather
  than infer a path across missing vertices. Other valid geometries remain usable,
  but the result is marked incomplete. Repair the source and rerun before treating
  its mileage as a complete project total.
- Calculation errors propagate into visible diagnostics instead of becoming
  successful zero-overlap results. Both GUI summaries show an incomplete notice.
  Excel marks incomplete totals and writes `Unavailable` for failed savings.
- While reviewing corridor output after removing repeated matches, fixed an
  existing polygon-join error that discarded earlier vertices. A regression
  checks that the corridor retains both ends of a straight section.

No new third-party dependencies were added. The large indentation change in
`overlap.py` comes from removing the old nested grouping loop; use `git diff -w`
to review the substantive changes without indentation noise.

## Verification completed

- `python -m pytest -q`: **79 passed**.
- `python -m compileall -q src`: passed.
- `git diff --check`: passed.
- [test_calculation_regressions.py](tests/test_calculation_regressions.py) contains
  35 new parameterized cases, including reverse digitization, three pipelines,
  unequal lengths, multipart minimums, exact mileage expectations, link cycles,
  path boundaries, malformed geometry, injected failures, Excel, and the GUI notice.
- Two existing short-path effective-length tests now explicitly set a minimum
  shorter than their fixtures. Their original geometry expectations remain; new
  tests separately verify that short paths receive no discount at the default minimum.
- Synthetic 10 km parallel-pair smoke check: 10 km bundled, 10 km savings, one
  section; roughly 0.09 seconds for length/search/section calculations in this
  environment. This is not a production-data performance benchmark.

## Morning review and remaining limits

1. Inspect the diff, especially the common-coverage and minimum-length policy.
2. Run representative project KMZs and compare their source lengths, qualifying
   overlap sections, summary savings, and exported workbook. No real project KMZs
   were available in the checked repository, so validation used controlled fixtures.
3. Visually inspect both normal and incomplete-result screens in the packaged GUI,
   and open exported corridors in Google Earth. Native GUI appearance, Windows
   packaging, macOS packaging/signing, and distribution CI were not exercised here.
4. Commit/review/merge when satisfied; no remote state was changed by this task.

This remains a sampled segment model, not an exact corridor-union or flight-route
optimizer. Segment length and midpoint alignment can affect endpoint estimates;
unsegmented trailing remainders receive no savings. Neighborhood allocations for
complex three-or-more-pipeline layouts remain approximate. Pairwise bundled rows
are not additive project savings; use the separately computed mileage-removed total.
The ECEF search tests do not certify dateline/polar corridor rendering, which still
uses the existing local polygon construction. Those are further review areas,
not claims of full geometric accuracy for every possible dataset.

Reference for archive links: [Google KML/KMZ relative references](https://developers.google.com/kml/documentation/kmzarchives#resolving_relative_references).
