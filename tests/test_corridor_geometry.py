import math
import xml.etree.ElementTree as ET

import pytest

from pipeline_calculator.export.geometry_validation import prepare_geometry, validated_ring
from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml


def section(points):
    return {'pipeline_1':'A','pipeline_2':'B','bundled_length_miles':1,'average_separation':2,
            'corridor_polygon':points}


@pytest.mark.parametrize('value',[float('nan'),float('inf'),181])
def test_invalid_polygon_without_valid_fallback_is_rejected(value):
    with pytest.raises(ValueError,match='No usable'):
        build_overlap_corridor_kml(section([(value,40),(-100,40),(-100,40.01),(value,40)]),1)


def test_bad_preferred_ring_uses_disclosed_valid_rectangle():
    data=section([(float('nan'),0),(1,0),(1,1)])
    data['oriented_polygon']=[(-100,40),(-99.999,40),(-99.999,40.001),(-100,40.001)]
    text=build_overlap_corridor_kml(data,1)
    assert 'oriented_rectangle' in text and 'Preferred geometry was invalid' in text
    assert 'nan' not in text
    ET.fromstring(text)


def test_degenerate_and_self_crossing_rings_and_invalid_center():
    with pytest.raises(ValueError):
        validated_ring([(0,0),(0,0),(0,0)])
    with pytest.raises(ValueError):
        validated_ring([(0,0),(0.001,0.001),(0,0.001),(0.001,0)])
    data=section([(0,0),(0.001,0),(0.001,0.001),(0,0)])
    data.update(center_lon=float('nan'),center_lat=0)
    with pytest.raises(ValueError,match='center'):
        prepare_geometry(data)


def test_serialized_rounding_cannot_collapse_ring():
    with pytest.raises(ValueError):
        validated_ring([(0,0),(1e-9,0),(0,1e-9)])


def test_independent_geometry_checker():
    from scripts.validation.geometry import ring_checks, inside
    square=[(0,0),(10,0),(10,10),(0,10),(0,0)]
    assert ring_checks(square)['self_intersections']==0
    assert inside((5,5),square) and not inside((50,5),square)
    assert ring_checks([(0,0),(10,10),(0,10),(10,0),(0,0)])['self_intersections']>0


def test_gallery_exports_are_valid_and_approximations_are_disclosed(tmp_path):
    import json
    from scripts.validation.build_gallery import run
    assert run(tmp_path)==0
    rows=json.loads((tmp_path/'report.json').read_text())['cases']
    assert {'right_angle','hairpin','loop','dateline','pole'} <= {row['fixture'] for row in rows}
    for row in rows:
        assert row['status']=='validated-approximation'
        assert row['serialized']['self_intersections']==0
        assert 'Approximate visualization' in row['description']
        if 'Geometry: oriented_rectangle' in row['description']:
            assert 'Rectangle approximation' in row['description']
