# Antero original KMZ import failure

October 2, 2026. Investigated against repository baseline `b65aad0e8751`.

## Reproduction and cause

The supplied original `AnteroMidstreamData.kmz` fails in `prepare_source()` with
`unsupported_archive_metadata`: "This archive uses unsupported entry metadata."
The first rejected member is `AnteroMidstreamData.kml`. The archive's SHA-256 is
`a35f8ba2dbc653749e3ec835cdab53479f2184f53c66e72d25527ef08a9d4012`.

All ten entries have Info-ZIP Unix ownership metadata (`0x7875`) followed by the
already supported extended timestamp field (`0x5455`). The ownership field is
`75780b000104000000000400000000`: version 1, four-byte UID 0, four-byte GID 0.
The application allowlist omitted this field, so preparation stopped before
inspecting KML. Direct ordinary parsing already succeeded before the change.

The [Info-ZIP field documentation in Apache Commons Compress](https://commons.apache.org/proper/commons-compress/apidocs/org/apache/commons/compress/archivers/zip/X7875_NewUnix.html)
identifies this as variable-width owner/group metadata. It does not supply
geometry or change member paths. Supporting its verified layout is an input
compatibility fix; a new geometry or XML repair rule is unnecessary.

The first supplied file, `Antero Midstream.kmz`, is a different archive
(`c5249c55e89a381815b9095734cf61859cf20b4005039d03873263c999b33650`).
It imported successfully before the change and did not reproduce the failure.

## Change and preservation

[Source acquisition](../../src/pipeline_calculator/parsers/source.py) now accepts
version 1 of `0x7875` only when both nonempty ID fields fit the declared payload
exactly. Unknown versions, truncation, trailing payload bytes, and unknown extra
field types remain rejected. Ownership bytes are retained without interpreting
or applying filesystem permissions. Existing special-file, path, CRC, size,
encryption, and compression checks remain in place. No archive is extracted.

The original is retained as
[a regression fixture](../../tests/fixtures/geography/antero_midstream_data.kmz),
with an [independent XML inventory and checksum](../../tests/fixtures/geography/antero_midstream_data.expected.json).
Preparation now reports `not_needed`, zero repair rules, and unchanged document
hashes. The original Downloads file was not modified.

The original KML is well-formed XML but has no XML namespace binding: it contains
a literal `<xmlns>` child instead. The existing ordinary parser supports
namespace-free input. This change does not rewrite or certify the document as
schema-valid KML.

## Verification

- 236 tests passed in 10.40 seconds: `test_antero_kmz.py`,
  `test_kml_repair_sources.py`, `test_kml_repair_engine.py`,
  `test_repair_integration.py`, `test_kmz_parsing.py`, and
  `test_kml_structure_safety.py`.
- The [real-file test](../../tests/test_antero_kmz.py) independently reads XML and
  compares every imported path and vertex, preserving ordering and multipart
  separation: 960 pipeline features, 1,113 paths, 231,376 line vertices, and
  7,275 points. All archive and document checksums are fixed expectations.
- Synthetic tests cover variable-width IDs, metadata on unused assets and
  directories, malformed lengths and versions, unknown fields, ordinary no-op
  input, and saving/reimporting an actual XML repair while retaining metadata.
- Default combined analysis through the desktop controller entry point completed
  in 31.73 seconds. Source length was 1,776,257.3752514147 meters
  (1,103.712955550303 US survey miles). This is an observed application result,
  not an independent overlap or length oracle. Settings and diagnostic counts
  are retained in [the run summary](antero-kmz-analysis.json).
- `git diff --check` passed. No packaged executable was rebuilt or tested;
  state breakdown and native GUI interaction were not exercised in this audit.

## Separate source and visualization warnings

The archive contains 8,238 Placemarks. Three have metadata but no geometry:
NALLEY (OBJECTID 7189), HEASTER TO PIONEER (11385), and 10885 (10885).
Their schemas describe facilities. The parser retains its existing warnings;
there are no coordinates from which to restore their locations safely. This
does not establish whether geometry was missing upstream or intentionally absent.
The change neither deletes these records nor invents locations.

The combined analysis also reported 1,007 `corridor_buffer_limit` and 1,007
`corridor_visualization_omitted` warnings. Some corridor maps exceeded existing
construction/self-overlap budgets. Mileage analysis completed, but full corridor
visualization did not. This is separate from the archive import failure and is
not resolved by accepting Unix ZIP metadata.
