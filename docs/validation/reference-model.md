# Independent reference model

The oracle in [intervals.py](../../tests/reference/intervals.py) imports only the
standard library. It never calls production segmentation, qualification, grouping,
or savings. Production comparison is a separate script.

Coordinates are Cartesian meters. Each original part has independent cumulative
distance. Positive finite-edge coverage is clipped to the other edge's longitudinal
extent and transverse strip; tangent orientation is undirected. The same test is
applied in both directions. Edge correspondences connect only when their intervals
touch on both paths. Unique interval unions must reach the minimum on both paths;
parts cannot combine to satisfy it. Endpoint contact has zero length. Numerical
epsilon is 1e-7 m, distinct from an operational acceptance tolerance.

For up to six common-axis lines, all disjoint mutually compatible group partitions
are enumerated on slabs bounded by original interval endpoints. Singletons are
allowed; savings minimize pass count on each slab. Production's greedy selection
is not reused. General bent/multipart geometry receives pairwise reference coverage;
there is no asserted optimum for arbitrary branches or turn correspondence.

Geographic fixtures use GRS80 azimuthal-equidistant placement, recording origin and
bearing. They are bounded to 10 km from their origin. All vertex-pair distances
are compared with ellipsoidal inverse distances; maximum distortion must stay
within 0.001 m + 1e-5 of the planar distance. This checks the fixture model, not the
operational accuracy of production mileage. Dateline/polar cases use a nearby origin.

The error report separates planar pair coverage, production pair coverage and
projection distortion. Optimal savings and production savings are separate fields.
Their **combined savings gap includes sampling and grouping**; it is not presented
as isolated heuristic loss. General curved multi-line optimum is explicitly
unsupported. Pairwise corridor-length sums are not project savings.
The report additionally unions intervals across qualified sections by original
part identity (`reference_unique_pair_meters`), so repeated branch correspondences
cannot inflate unique coverage. The section-sum comparison and unique-coverage
comparison are separate: they answer different questions.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_reference.py
.\.venv\Scripts\python.exe scripts/validation/compare_reference.py --output .validation-output/reference
```

Analytical tests cover identical/offset/end-contact lines, incompatible chains,
mutually compatible groups, reversal, partial coverage, duplicate union and multipart
minimum handling. Deliberately wrong counting/tail calculations are rejected.
The sweep covers 280 combinations of geometry, resolution, orientation and thresholds.
Operational tolerances and any changed business rule remain runbook R2.
