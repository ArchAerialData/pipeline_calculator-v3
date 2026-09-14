"""Deliberately high workload advisories, separate from hard processing limits."""
from __future__ import annotations

import random
import math
from pipeline_calculator.core.segmentation import MAX_ANALYSIS_SEGMENTS

# Density is retained as bounded diagnostic sampling. Advisory decisions use
# measured runtime projections in ExecutionContext, not these work counts.
DENSITY_SAMPLE_MIN_WORK = 10_000_000
DENSITY_SAMPLE_SIZE = 256


class WorkloadWarning(str):
    """Text-compatible notice with explicit sections for GUI presentation."""
    introduction = 'This analysis is expected to take longer than one minute at the current settings.'
    impact = 'Overlap analysis may take longer and use substantial memory. Progress may pause during intensive calculations.'
    recommendation = 'Split the geometry into separate KML/KMZ files, or simplify a copy while preserving the accuracy you need.'
    decision = 'You can cancel now or continue with the current settings. Safety limits remain active.'

    def __new__(cls, detail):
        notice = super().__new__(cls, '\n\n'.join((cls.introduction, detail, cls.impact,
                                                  cls.recommendation, cls.decision)))
        notice.detail = detail
        return notice


def warning_text(detail):
    return WorkloadWarning(detail)


def runtime_warning(estimated_seconds, elapsed_seconds):
    detail = (f'Processing has exceeded 60 seconds ({elapsed_seconds:.0f} s so far).'
              if elapsed_seconds > 60 else
              f'Estimated total processing time: about {math.ceil(estimated_seconds):,} seconds.')
    return warning_text(detail+' Based on processing speed observed in this run; remaining time may vary.')


def check_segment_workload(estimated_segments, context):
    if context is not None:
        context.estimated_segments = estimated_segments


def check_density_workload(tree, points, radius, context):
    """Bounded, repeatable stratified neighbor sample; never materialize pairs.

    Counts include self/same-pipeline entries, matching the raw visit budget,
    not the unique cross-pipeline check budget. A sample can miss a small hotspot;
    both hard limits remain active.
    """
    count = len(points)
    if (context is None or not context.interactive or context.workload_accepted
            or count * count < DENSITY_SAMPLE_MIN_WORK):
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
    return estimate
