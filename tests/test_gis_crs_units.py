"""Geographic source bounds follow declared angular units, not degree guesses."""
import pytest
from pyproj import CRS, Transformer

from pipeline_calculator.importing.coordinates import CoordinateOperation
from pipeline_calculator.importing.models import ImportFailure


def test_grad_geographic_source_is_valid_beyond_degree_numeric_bounds():
    source = CRS.from_wkt('GEOGCRS["WGS 84 grads",DATUM["World Geodetic System 1984",'
        'ELLIPSOID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'
        'CS[ellipsoidal,2],AXIS["longitude",east,ORDER[1]],AXIS["latitude",north,ORDER[2]],'
        'ANGLEUNIT["grad",0.015707963267948967]]')
    points = [(195, 95), (196, 95)]
    expected = [Transformer.from_crs(source, 4326, always_xy=True).transform(*point) for point in points]
    actual = CoordinateOperation(source).path(points)
    assert actual[0] == pytest.approx((175.5, 85.5))
    assert actual[1] == pytest.approx((176.4, 85.5))
    assert actual == expected


def test_degree_source_out_of_range_latitude_is_not_wrapped_or_joined():
    with pytest.raises(ImportFailure) as error:
        CoordinateOperation('EPSG:4326').path([(-100, 30), (-100, 95), (-99, 30)])
    assert error.value.code == 'invalid_geometry'
