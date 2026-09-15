"""Compatibility API for ordinary corridor previews and validation callers."""
from pipeline_calculator.core import corridor_geometry as _geometry

GEOD = _geometry.GEOD
TOPOLOGY_PAIR_BUDGET = _geometry.TOPOLOGY_PAIR_BUDGET
MAX_RING_POINTS = _geometry.MAX_RING_POINTS
coordinate = _geometry.coordinate


def validated_ring(points):
    return _geometry.validated_ring(points, max_points=MAX_RING_POINTS,
                                    topology_budget=TOPOLOGY_PAIR_BUDGET)


def prepare_geometry(section):
    # Keep module-level overrides usable by existing integrations and tests.
    return _geometry.prepare_geometry(section, validator=validated_ring)
