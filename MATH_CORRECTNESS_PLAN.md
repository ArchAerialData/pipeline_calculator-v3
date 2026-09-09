# Math/Algorithm Correctness Plan (Critical Fixes)

This document breaks down three **correctness-impacting** issues in the current overlap-analysis pipeline and proposes a phased plan to fix each, with concrete file changes, acceptance criteria, and proof (math + tests + reproducible diagnostics).

Scope (current repo):
- Core logic lives under `src/pipeline_calculator/core/*` and is used by both:
  - the modular GUI (`src/pipeline_calculator/gui/*`)
  - the legacy monolith (`src/pipeline_calculator_v3.py`) via delegation wrappers

That means fixes in `src/pipeline_calculator/core/*` automatically improve both the modular and legacy implementations.

Dependency policy for these fixes:
- **Phases 1-3 require no new third-party dependencies** beyond what is already present in `requirements.txt` today (notably `pyproj`, `numpy`, `scipy`).
- If a later optimization removes SciPy, that will be called out explicitly and `requirements.txt` can be updated accordingly.

Status (implementation in this repo):
- Phase 1 (Segmentation): **COMPLETED** (2026-02-09)
- Phase 2 (Spatial indexing in meters): **COMPLETED** (2026-02-09)
- Phase 3 (Anti-parallel bearing handling): **COMPLETED** (2026-02-09)

---

## Baseline Evidence (Current Issues Are Real)

These observations were confirmed on the included sample data:
- `test_data/Brazos_NGL and Delaware_Gas combined.kmz`

### Reproduce Baseline Locally
Run from repo root:

1) Inspect vertex-to-vertex edge lengths (meters):
```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from __future__ import annotations

from pathlib import Path
import statistics
from pyproj import Geod

from pipeline_calculator.parsers.kml_kmz import extract_features_from_file

path = Path("test_data/Brazos_NGL and Delaware_Gas combined.kmz")
pipelines, _ = extract_features_from_file(str(path))
g = Geod(ellps="GRS80")

all_dists = []
for p in pipelines:
    coords = p["coordinates"]
    for (lon1, lat1), (lon2, lat2) in zip(coords, coords[1:]):
        _, _, d = g.inv(lon1, lat1, lon2, lat2)
        all_dists.append(abs(float(d)))

all_dists.sort()
print("pipelines:", len(pipelines))
print("edges:", len(all_dists))
print("min_m:", all_dists[0])
print("p50_m:", all_dists[len(all_dists)//2])
print("p90_m:", all_dists[int(len(all_dists)*0.9)])
print("p99_m:", all_dists[int(len(all_dists)*0.99)])
print("max_m:", all_dists[-1])
print("mean_m:", statistics.mean(all_dists))
PY
```

2) Inspect segmentation spacing on a “long-edge” pipeline (this output is the *proof* that Phase 1 is needed, and should improve after Phase 1 is completed):
```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from __future__ import annotations

from pathlib import Path
import statistics
from pyproj import Geod

from pipeline_calculator.parsers.kml_kmz import extract_features_from_file
from pipeline_calculator.core.segmentation import segment_pipeline

path = Path("test_data/Brazos_NGL and Delaware_Gas combined.kmz")
pipelines, _ = extract_features_from_file(str(path))
target = next((p for p in pipelines if p.get("name") == "Item_8"), pipelines[0])

geod = Geod(ellps="GRS80")
seg_len = 5.0
segments = segment_pipeline(geod, target["coordinates"], seg_len)

pts = [s["midpoint"] for s in segments]
dists = []
for (lon1, lat1), (lon2, lat2) in zip(pts, pts[1:]):
    _, _, d = geod.inv(lon1, lat1, lon2, lat2)
    dists.append(abs(float(d)))
dists.sort()

print("target:", target.get("name"), "coords:", len(target["coordinates"]))
print("segments:", len(segments))
print("spacing_m: min/p50/p90/p99/max/mean")
print(
    dists[0],
    dists[len(dists)//2],
    dists[int(len(dists)*0.9)],
    dists[int(len(dists)*0.99)],
    dists[-1],
    statistics.mean(dists),
)
PY
```

### 1) Vertex-to-vertex edges can be kilometers long
Across that file, the maximum KML edge length is about **6,034m**.

Why it matters:
- Any segmentation logic must handle very long edges correctly, because each long edge produces thousands of 5m analysis segments.

### 2) Current segmentation output spacing is not “~segment_length”
On pipeline `Item_8` in that same file (using `segment_length=5m`), the distances between consecutive returned “midpoints” had:
- median ~ **0m** (should be ~5m)
- max > **100m** (should be ~5m)

This directly impacts:
- which segments are considered “near” (KDTree candidates)
- continuity / bundled section grouping
- effective length computation (cluster participation)

---

## Phase 1: Fix Segmentation (Correct Fixed-Length Segment Placement)

### Problem Statement
File: `src/pipeline_calculator/core/segmentation.py`

The existing `segment_pipeline()` logic is incorrect for long edges because it updates the current point inside the `while accumulated_distance >= segment_length` loop without recomputing the remaining edge geometry. This causes segment points to “cluster” and then “jump”, instead of being spaced approximately `segment_length` along the polyline.

### Fix Strategy (Algorithm)
Rewrite segmentation to explicitly “walk” along the polyline and create fixed-length segments.

Key design goals:
1. Produce segment records with:
   - `midpoint`: a representative point for neighbor searching
   - `bearing`: a representative direction for parallel checks
   - `length`: `segment_length` (meters)
   - `segment_index`: stable sequential index
2. Correctly handle:
   - very long edges (km)
   - very short edges (< segment_length) by carrying the remainder into the next edge

#### Proposed approach (walk along edges with carry)
Maintain:
- `seg_start`: start point of the current analysis segment (initially first coordinate)
- `remaining_m`: meters remaining to complete the current analysis segment (initially `segment_length`)

For each polyline edge `(edge_start -> edge_end)`:
1. Compute `az, _, edge_dist = geod.inv(edge_start, edge_end)`
2. While `edge_dist >= remaining_m`:
   - Compute `seg_end` at `remaining_m` along the edge geodesic:
     - `seg_end = geod.fwd(edge_start, az, remaining_m)`
   - Define segment bearing using `geod.inv(seg_start, seg_end)` (orientation of the segment)
   - Define segment midpoint using `geod.fwd(seg_start, seg_bearing, segment_length/2)`
   - Append segment record (length always `segment_length`)
   - Reset:
     - `seg_start = seg_end`
     - `remaining_m = segment_length`
     - `edge_start = seg_end` (continue on the remainder of the same edge)
   - Recompute `(az, edge_dist)` for the remainder to stay on the original geodesic to `edge_end`
3. If `edge_dist < remaining_m`:
   - Consume the full edge: `remaining_m -= edge_dist`
   - Move to next polyline edge

Notes:
- This creates fixed-length segments along the *polyline path length*, not “fraction of vertex spacing”.
- It eliminates the 0m spacing / 100m gaps observed in baseline diagnostics.

### Code Changes (Paths)
Edits:
- `src/pipeline_calculator/core/segmentation.py`

Potential helper addition (optional, for clarity):
- `src/pipeline_calculator/core/geom.py`
  - `geodesic_point_at(geod, lon, lat, az, dist_m) -> (lon, lat)`

### Dependencies / Requirements
- No new third-party dependencies are required for this phase.
- Uses `pyproj` (already in `requirements.txt`).

### Proof / Evidence (Math + Tests + Diagnostics)
Math basis:
- `pyproj.Geod.inv()` gives distance (m) and forward azimuth (deg) between two lon/lat points on the ellipsoid.
- `pyproj.Geod.fwd()` gives the point reached after traveling a given distance (m) along the geodesic with a given azimuth.
- Repeatedly consuming `remaining_m` guarantees segment boundaries are spaced by exact path-length increments (subject only to floating point tolerance).

Tests to add/update:
1. New: `tests/test_segmentation_spacing.py`
   - Create a synthetic polyline consisting of a *single long edge* (e.g., ~10,000m).
   - Segment with `segment_length=5m`.
   - Assert:
     - number of segments ~= floor(total_len/segment_length)
     - distance between consecutive segment midpoints is not near-zero and not huge
       - For midpoints, expected spacing is ~`segment_length` (within a reasonable tolerance).
2. Update existing test:
   - `tests/test_segmentation.py` should validate not just count/index, but also spacing sanity.

Reproducible diagnostic command (non-test; used as human proof):
- Add script: `tests/tools/segmentation_diagnostics.py` (or `scripts/dev/segmentation_diagnostics.py`)
  - Runs segmentation on `test_data/*.kmz` and prints spacing distribution (min/p50/p90/p99/max).

Acceptance criteria:
- For synthetic long-edge input: consecutive segment midpoints are approximately spaced (no median ~0, no max >> segment_length).
- Unit test suite remains green.

Implementation notes (what we actually changed in this repo):
- Updated: `src/pipeline_calculator/core/segmentation.py`
- Added test: `tests/test_segmentation_spacing.py`

Post-fix diagnostic (2026-02-09) on `test_data/Brazos_NGL and Delaware_Gas combined.kmz` (`Item_8`, `segment_length=5m`):
- segments: 3,852
- inter-midpoint spacing (meters): min=3.613, p50=5.000, p90=5.000, p99=5.000, max=5.000, mean=4.998

---

## Phase 2: Fix Spatial Neighbor Search (Index in Meters, Not Degrees)

### Problem Statement
Files:
- `src/pipeline_calculator/core/overlap.py`
- `src/pipeline_calculator/core/effective_length.py`

Both use a KDTree built on raw `(lon, lat)` degrees and query with:
- `detection_range_deg = detection_range_m / 111000`

This is **incorrect** because:
- 1 degree of latitude ≈ 111,320m (varies slightly)
- 1 degree of longitude ≈ 111,320m * cos(latitude)

So a single “meters -> degrees” conversion is only valid for latitude, not longitude.
Result:
- The search radius in longitude is too small by a factor of `cos(latitude)`.
- This can cause **false negatives** in candidate selection, meaning overlaps can be missed entirely.

### Fix Strategy (Math-Backed)
Index in a local XY coordinate system measured in meters, then query by radius in meters.

Recommended projection (simple + sufficient for our use case):
- Equirectangular approximation around an origin `(lon0, lat0)`:
  - `x_m = (lon - lon0) * 111320 * cos(lat0)`
  - `y_m = (lat - lat0) * 111320`

Why this works here:
- Our neighbor radius is small (e.g., 15m).
- We only need local correctness for candidate generation; we still verify true distance with `geod.inv()`.
- This avoids missing east-west neighbors at mid/high latitudes.

### Code Changes (Paths)
New:
```text
src/pipeline_calculator/core/spatial.py
  - compute_origin(points_lonlat) -> (lon0, lat0)
  - lonlat_to_xy(lon, lat, lon0, lat0) -> (x_m, y_m)
  - lonlat_array_to_xy(points, lon0, lat0) -> np.ndarray shape (N,2)
```

Edits:
- `src/pipeline_calculator/core/overlap.py`
  - Build KDTree on XY meters, not lon/lat degrees
  - Query with `radius_m = detection_range`
  - Keep the existing `geod.inv()` exact distance check as final gate
- `src/pipeline_calculator/core/effective_length.py`
  - Same fix: KDTree in meters + radius in meters

### Dependencies / Requirements
- No new third-party dependencies are required for this phase.
- Uses `numpy` and `scipy` (already in `requirements.txt`).
- (Optional future) if we later add a numpy-only grid index to remove SciPy, that change will include:
  - explicit doc updates
  - `requirements.txt` update to make SciPy optional or remove it

### Proof / Evidence (Math + Tests)
Math evidence (why current approach fails):
- At latitude `phi`, 1 degree of longitude is `111320*cos(phi)` meters.
- If we use `detection_range_deg = R/111000`, then the implied longitude radius in meters is:
  - `R_lon ≈ detection_range_deg * 111320*cos(phi) ≈ R * cos(phi)`
- Example at 60 degrees latitude:
  - `cos(60) = 0.5`
  - a 15m intended radius becomes ~7.5m in longitude in the KDTree candidate search
  - neighbor candidates at 10–15m east-west can be missed

Tests to add:
1. `tests/test_spatial_index_high_latitude.py`
   - Construct two “segment midpoints” at latitude 60 degrees separated by 12m in longitude direction.
   - With `detection_range=15m`, the corrected XY KDTree must return them as candidates.
   - (Optionally) assert that the old degrees-based query would *not* include them (documenting the regression).
2. `tests/test_effective_length_neighbors.py`
   - Synthetic pipelines with known overlaps at non-equatorial latitudes; ensure effective length logic sees the neighbor.

Acceptance criteria:
- High-latitude synthetic test passes (no false negatives in candidate generation).
- Results on existing `test_data/` inputs do not regress (may increase detected overlaps, which is expected).

Implementation notes (what we actually changed in this repo):
- Added: `src/pipeline_calculator/core/spatial.py`
- Updated: `src/pipeline_calculator/core/overlap.py`
- Updated: `src/pipeline_calculator/core/effective_length.py`
- Added tests:
  - `tests/test_spatial_index_high_latitude.py` (end-to-end overlap neighbor detection at 60 deg lat)
  - `tests/test_effective_length_high_latitude.py` (end-to-end cluster discount at 60 deg lat)

---

## Phase 3: Handle Anti-Parallel Lines (Bearing 0 vs 180 Should Count as Parallel)

### Problem Statement
Files:
- `src/pipeline_calculator/core/overlap.py`
- `src/pipeline_calculator/core/effective_length.py`

Current “parallel” test uses:
- `bearing_diff = min(abs(b1-b2), 360-abs(b1-b2))`
- `bearing_diff <= angular_tolerance`

This rejects anti-parallel segments:
- Example: `b1=0`, `b2=180` => `bearing_diff=180` => rejected

But in GIS data, the same physical pipeline can be digitized in opposite directions, so this causes missed overlaps.

### Fix Strategy (Use Orientation, Not Direction)
Treat “parallel” as line *orientation* difference, which is modulo 180 degrees.

Compute:
1. `diff = abs(b1 - b2) % 360`
2. `diff = min(diff, 360 - diff)`  (now diff is in `[0, 180]`)
3. `orientation_diff = min(diff, 180 - diff)`  (now in `[0, 90]`)

Then:
- parallel if `orientation_diff <= angular_tolerance`

Examples:
- `b1=0`, `b2=180`
  - `diff=180`
  - `orientation_diff=min(180,0)=0` => parallel
- `b1=10`, `b2=190` => also parallel
- `b1=0`, `b2=170`
  - `diff=170`
  - `orientation_diff=10` => parallel if tolerance >=10

### Code Changes (Paths)
New helper (recommended, shared by both modules):
```text
src/pipeline_calculator/core/angles.py
  - bearing_orientation_diff(b1_deg: float, b2_deg: float) -> float
```

Edits:
- `src/pipeline_calculator/core/overlap.py`
  - Replace current bearing check with `orientation_diff` check
- `src/pipeline_calculator/core/effective_length.py`
  - Same replacement

### Dependencies / Requirements
- No new third-party dependencies are required for this phase.
- Uses only stdlib math plus existing core modules.

### Proof / Evidence (Tests)
Tests to add:
- `tests/test_parallel_bearing_orientation.py`
  - Validate that bearings `0` and `180` are treated as parallel.
  - Validate that `0` and `90` are not parallel for typical tolerances.

Acceptance criteria:
- New bearing orientation tests pass.
- Existing overlap tests remain green, and anti-parallel cases become detectable.

Implementation notes (what we actually changed in this repo):
- Added: `src/pipeline_calculator/core/angles.py`
- Updated: `src/pipeline_calculator/core/overlap.py`
- Updated: `src/pipeline_calculator/core/effective_length.py`
- Added tests:
  - `tests/test_parallel_bearing_orientation.py` (unit tests for orientation math)
  - `tests/test_parallel_antiparallel_detection.py` (end-to-end: same line digitized opposite directions)

---

## Rollout / Validation Checklist (After Each Phase)

For each phase:
1. `PYTHONPATH=src .venv/bin/python -m pytest`
2. Run a small diagnostic on the real KMZ:
   - segmentation spacing stats (Phase 1)
   - overlap count / savings deltas (Phases 2-3)
3. Manual GUI smoke (fast):
   - import KMZ, verify results render, export XLSX works, open corridor KML works

Expected behavior changes:
- After Phase 1: overlaps/effective length may change (because segment geometry is now correct).
- After Phase 2: overlaps should become **more complete** (fewer false negatives).
- After Phase 3: overlaps should become **more complete** when pipelines are digitized in opposite directions.
