import pytest
from pyproj import Geod
from pipeline_calculator.core.coordinates import segment_pipeline_paths
from pipeline_calculator.core.corridor_coverage import qualified_path_runs


GEOD = Geod(ellps='GRS80')


def data():
    a = (-100, 40)
    b = GEOD.fwd(*a, 0, 12)[:2]
    c = GEOD.fwd(*b, 90, 15)[:2]
    records = [{'id': index, 'name': 'Duplicate name', 'coordinate_paths': [[a, a, b, c], [c, b, a]]}
               for index in range(2)]
    for record in records:
        record['segments'] = segment_pipeline_paths(GEOD, record, 5)
    return records, dict(pair=(0, 1), paths=(0, 0), segment_ids=({1, 2, 4}, {1, 2, 4}))


def test_disjoint_qualified_runs_keep_bends_and_exclude_gaps_and_tails():
    records, section = data()
    runs = qualified_path_runs(records, section, 5, GEOD)
    assert [(r.start_m, r.end_m) for r in runs] == [(5, 15), (20, 25)] * 2
    assert records[0]['coordinate_paths'][0][2] in runs[0].coordinates
    assert runs[0].source_id != runs[2].source_id
    for run in runs:
        measured = sum(GEOD.inv(*a, *b)[2] for a, b in zip(run.coordinates, run.coordinates[1:]))
        assert measured == pytest.approx(run.end_m - run.start_m, abs=1e-8)


@pytest.mark.parametrize('bad', [-1, 500, True, 1.5])
def test_invalid_segment_indices_are_rejected(bad):
    records, section = data()
    section['segment_ids'] = ({bad}, {1})
    with pytest.raises(ValueError, match='invalid segment'):
        qualified_path_runs(records, section, 5, GEOD)


def test_indices_from_other_coordinate_paths_cannot_create_connector():
    records, section = data()
    section['segment_ids'] = ({5}, {1})
    with pytest.raises(ValueError, match='another path'):
        qualified_path_runs(records, section, 5, GEOD)


def test_invalid_chainage_is_not_silently_clamped():
    records, section = data()
    records[0]['segments'][1]['path_segment_index'] = 100
    section['segment_ids'] = ({1}, {1})
    with pytest.raises(ValueError, match='exceeds'):
        qualified_path_runs(records, section, 5, GEOD)


def test_state_path_indices_are_explicitly_scope_local():
    records, section = data()
    runs = qualified_path_runs(records, section, 5, GEOD, scope='TX')
    assert runs[0].provenance()['scope'] == 'TX'
    assert 'scope_path_index' in runs[0].provenance()
    assert 'original_path_index' not in runs[0].provenance()


def test_extraction_checks_chainage_allocation_before_copying_source(monkeypatch):
    from pipeline_calculator.core import corridor_coverage
    records, section = data()
    def forbidden(*args, **kwargs):
        raise AssertionError('Oversized path must be rejected before chainage allocation')
    monkeypatch.setattr(corridor_coverage, 'MeasuredPath', forbidden)
    with pytest.raises(ValueError, match='extraction budget'):
        qualified_path_runs(records, section, 5, GEOD, max_points=3)


def test_extraction_checks_aggregate_run_coordinates_before_span(monkeypatch):
    from pipeline_calculator.core import corridor_coverage
    records, section = data()
    original = corridor_coverage.MeasuredPath.span
    calls = 0
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(corridor_coverage.MeasuredPath, 'span', counted)
    with pytest.raises(ValueError, match='extraction budget'):
        qualified_path_runs(records, section, 5, GEOD, max_points=5)
    assert calls == 2


def test_extraction_charges_new_cache_against_shared_job_budget():
    from pipeline_calculator.core.corridor_buffer import CorridorGeometryBudget
    records, section = data()
    budget = CorridorGeometryBudget(max_work_vertices=5)
    with pytest.raises(ValueError, match='job construction budget'):
        qualified_path_runs(records, section, 5, GEOD, budget=budget)
