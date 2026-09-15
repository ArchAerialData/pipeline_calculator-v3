# Deterministic input generator

Run the suite's regeneration command from the [suite README](../README.md) to
also rebuild independent expectations. To regenerate only the input archives and
previews, from the repository root run:

```powershell
python tests/fixtures/geography/pipeline_kmz_regression_suite/generator/generate.py
```

`--output PATH` writes the same organized `fixtures/`, `previews/`, and
`validation/design_manifest.json` layout elsewhere. `--no-previews` skips rendering.
Generation operates offline and imports no application module.

The generator verifies the complete bundled boundary ZIP and every state WKB
member against the retained manifest. It uses those existing coordinates without
transforming or simplifying boundaries. [The design manifest](../validation/design_manifest.json)
records the resource provenance, baseline, seed, version, canonical boundary ring
edge indices, source identities, and intended assertions. Those intentions become
accepted expectations only after the separate references verify the saved KMZ.

Coordinates are serialized to 14 decimal places. ZIP timestamp, entry name,
compression settings, creator platform, and permissions are fixed. Input KMZs
contain only actual source LineStrings, preserving disconnected paths within a
source as MultiGeometry. Preview boundaries and annotations are separate PNGs.
Varying altitude values intentionally exercise two-dimensional mileage accounting.

## Variants

Default generation also writes these archives in `fixtures/variants/`:

- `01_three_state_disconnected_networks__source_order.kmz`: seeded source-order
  shuffle; names, keys, OBJECTIDs, XML IDs, and geometry remain attached to each source.
- `04_shared_border_and_near_border__reversed.kmz`: reverses every path independently.
- `04_shared_border_and_near_border__redundant_vertices.kmz`: inserts vertices on
  each original GRS80 geodesic, retaining each existing endpoint and path break.

The shared meridian's longitude is exactly the native `-103.064732`.
The transverse fixture's endpoint touch uses a native vertex: serializing an
interpolated point on an oblique edge can move it a fraction of a nanometre into
an interior. Reversal and extra vertices preserve geographic allocation; sampled
savings are recomputed independently because reversal can move a trailing remainder.

## Larger profile

```powershell
python tests/fixtures/geography/pipeline_kmz_regression_suite/generator/generate.py --profile stress --stress-groups 24 --output PATH
```

This creates 24 spatially separated branching groups, each containing a close
three-source corridor and a counted branch, under `fixtures/stress/`. The parameter
changes meaningful overlap work, rather than padding bytes. Generating this file
does not establish successful application analysis; the suite's separate stress
measurement records actual segments, results, runtime, and memory when run.

## Preview interpretation

The four overview PNGs show the real geographic layout. Three extra images under
`previews/details/` expose meter and centimeter controls with independently scaled
east/north axes. Their explicit scale labels prevent interpreting the visual
exaggeration as physical corridor width. Black dashed borders are the canonical
resource geometry and never appear as input pipelines.
