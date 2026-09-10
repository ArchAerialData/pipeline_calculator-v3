# Corpus manifest v1

Run `python scripts/validation/run_corpus.py --manifest PATH --private-root ROOT
--output OUTPUT`. Use `--strict` when all specified inputs must be available.
Default `--suite local` generates synthetic files and characterizes the existing
tracked KMZ without asserting its operational correctness. Output is local JSON,
Markdown, and synthetic-only JSON/XLSX exports. No data is uploaded.

Copy this template and fill real hashes/expectations before use:

```json
{
  "schema_version": 1,
  "fixtures": [{
    "id": "project-id",
    "path": "relative-to-private-root.kmz",
    "sha256": "REPLACE_WITH_SHA256",
    "purpose": "Representative multipart job",
    "status": "unreviewed",
    "parameters": {},
    "expectations": {},
    "basis": null,
    "reviewer": null,
    "provenance": null,
    "required": false
  }]
}
```

Statuses are `analytic`, `unreviewed`, and `reviewed`. Reviewed entries require a
reviewer. Assertions require a basis; numeric expectations use
`{"value": 300, "absolute_tolerance": 5}`. Comparable fields are `complete`,
`pipelines`, `original_meters`, `savings_meters`, `sections`, and `diagnostics`.
Use `expected_error: true` for authored malformed inputs. Unknown expectations
remain empty. Current app output is not an independent expectation.

Reports distinguish failures, missing optional input and unreviewed results.
Hash mismatch always fails. Private files stay outside Git; reports contain IDs,
hashes, aggregate metrics and diagnostic codes, not source names or geometry.
Keep reports private too if aggregate project statistics are confidential.
