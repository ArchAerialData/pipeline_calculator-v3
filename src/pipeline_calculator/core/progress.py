"""Measured work progress and conservative projections from this run's speed."""
import math

WARNING_DURATION_SECONDS = 60.0
MIN_RUNTIME_SAMPLE_SECONDS = 3.0
RUNTIME_CONFIRM_SECONDS = 1.0

# Fixed shares of the processing workflow, advanced only by real work counts or
# completed stages. Percent is work progress, not a promise of remaining time.
STAGES = {
    'Reading documents': ('reading', 0.00, .05),
    'Calculating source lengths': ('lengths', .05, .10),
    'Segmenting paths': ('segmentation', .10, .35),
    'Segmenting path': ('segmentation', .10, .35),
    'Building spatial index': ('index', .35, .40),
    'Checking input density': ('density', .40, .41),
    'Searching neighbors': ('search', .41, .82),
    'Qualifying sections': ('qualification', .82, .85),
    'Building corridors': ('corridors', .85, .90),
    'Building group graph': ('graph', .90, .93),
    'Sorting group candidates': ('sorting', .93, .94),
    'Calculating savings': ('savings', .94, .99),
    'Finalizing results': ('finalizing', .99, .99),
    'Complete': ('complete', 1.0, 1.0),
}


class WorkProgress:
    def __init__(self):
        self.fraction = 0.0

    def update(self, stage, completed, total):
        if stage in STAGES:
            _, start, end = STAGES[stage]
            part = min(1.0, max(0.0, completed/total)) if total and total > 0 else 0.0
            self.fraction = max(self.fraction, start+(end-start)*part)
        return self.fraction


class RuntimeProjection:
    """Warn only when observed work projects beyond a minute.

    Project unfinished work in the current stage only; do not invent a speed for
    unstarted stages. A sustained estimate avoids single slow startup samples.
    This can warn partway through processing and cannot predict every later cost.
    """
    def __init__(self):
        self.stage = None
        self.stage_started = 0.0
        self.over_since = None
        self.previous_completed = None

    def observe(self, stage, completed, total, elapsed):
        phase = STAGES.get(stage, (stage,))[0]
        if stage in ('Waiting for your choice', 'Finalizing results', 'Complete'):
            return None
        if self.stage != phase:
            self.stage, self.stage_started = phase, elapsed
            self.over_since, self.previous_completed = None, None
        if elapsed > WARNING_DURATION_SECONDS:
            return elapsed
        spent = elapsed-self.stage_started
        if not total or completed <= 0 or completed >= total or spent < MIN_RUNTIME_SAMPLE_SECONDS:
            return None
        projected = elapsed+(total-completed)*spent/completed
        if not math.isfinite(projected) or projected <= WARNING_DURATION_SECONDS:
            self.over_since = None
        elif self.over_since is None:
            self.over_since, self.previous_completed = elapsed, completed
        elif elapsed-self.over_since >= RUNTIME_CONFIRM_SECONDS and completed > self.previous_completed:
            return projected
        return None
