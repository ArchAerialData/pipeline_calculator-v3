"""Deliberately high workload advisories, separate from hard processing limits."""
from __future__ import annotations

import random
from pipeline_calculator.core.segmentation import MAX_ANALYSIS_SEGMENTS

# Segment advisory is near the existing 1M cap; sampled density must exceed
# twice the 5M candidate cap to avoid warning on merely slow successful jobs.
# These are work estimates, not promised runtimes. Tune from real evidence.
SEGMENT_WARNING_THRESHOLD = 750_000
CANDIDATE_WARNING_THRESHOLD = 10_000_000
DENSITY_SAMPLE_SIZE = 256


def warning_text(detail):
    return ("This file appears exceptionally large or densely grouped at the selected settings.\n\n"
            + detail + "\n\nOverlap analysis may take a very long time, use substantial memory, "
            "or appear stalled. Consider splitting the geometry into smaller KML/KMZ files, "
            "or simplifying a copy if that preserves the distance accuracy you need.\n\n"
            "Cancel to stop now, or Continue anyway. Existing safety limits still apply.")


def check_segment_workload(estimated_segments, context):
    if context is not None and estimated_segments >= SEGMENT_WARNING_THRESHOLD:
        detail = (f"More than {MAX_ANALYSIS_SEGMENTS:,} analysis segments would be generated; "
                  "overlap exceeds the supported size. Continuing will retain source mileage "
                  "with an incomplete-overlap notice." if estimated_segments > MAX_ANALYSIS_SEGMENTS else
                  f"About {estimated_segments:,} analysis segments would be generated.")
        context.confirm_workload(warning_text(detail))


def check_density_workload(tree, points, radius, context):
    """Bounded, repeatable stratified neighbor sample; never materialize pairs.

    Counts include self/same-pipeline entries because the existing search budget
    includes them too. A sample can miss a small hotspot; hard limits remain active.
    """
    count = len(points)
    if (context is None or not context.interactive or context.workload_accepted
            or count * count < CANDIDATE_WARNING_THRESHOLD):
        return
    context.report('Checking input density', 0, min(count, DENSITY_SAMPLE_SIZE))
    samples = min(count, DENSITY_SAMPLE_SIZE)
    estimate = 0
    rng = random.Random(0x50495045)
    for i in range(samples):
        context.check()
        # Jitter within each stratum to avoid aliasing repeated pipeline layouts.
        # Weight by stratum size, including a shorter final stratum.
        start, stop = i * count // samples, (i + 1) * count // samples
        index = rng.randrange(start, stop)
        estimate += int(tree.query_ball_point(points[index], radius, return_length=True)) * (stop-start)
        context.report('Checking input density', i + 1, samples)
    if estimate >= CANDIDATE_WARNING_THRESHOLD:
        context.confirm_workload(warning_text(
            f"A sample suggests about {estimate:,} nearby-segment checks. "
            "This is an estimate, not a predicted completion time."))
