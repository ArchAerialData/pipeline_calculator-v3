import ast
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parent))
from reference import intervals as ref


@pytest.mark.parametrize('lines,minimum,expected',[
    ([(0,0,300),(0,0,300)],200,300),
    ([(0,0,300),(2,50,350)],200,250),
    ([(0,0,300),(2,50,350)],251,0),
    ([(0,0,300),(0,300,600)],200,0),
    ([(0,0,300),(2,0,300),(4,0,300)],200,600),
    ([(0,0,300),(10,0,300),(20,0,300)],200,300),
    ([(0,300,0),(2,350,50)],200,250),
])
def test_analytical_group_cases(lines,minimum,expected):
    assert ref.common_axis_savings(lines,minimum=minimum)==pytest.approx(expected,abs=1e-7)


def test_bilateral_partial_intervals_and_multipart_minimum():
    a=[[(0,0),(0,300)]];b=[[(2,50),(2,350)]]
    sections=ref.pair_coverage(a,b)
    assert sections[0]['intervals_a']==[(50,300)]
    assert sections[0]['intervals_b']==[(0,250)]
    assert not ref.pair_coverage([[(0,0),(0,150)],[(0,400),(0,550)]],
                                 [[(2,0),(2,150)],[(2,400),(2,550)]])


def test_intervals_and_deliberate_regressions_are_rejected():
    assert ref.union([(0,10),(5,20),(20,20.5)])==[(0,20.5)]
    assert ref.subtract([(0,30)],[(5,10),(20,40)])==[(0,5),(10,20)]
    assert ref.intersection([(0,20)],[(10,30)])==[(10,20)]
    def check_length(function):
        assert function([(0,10),(5,20),(20,20.5)])==pytest.approx(20.5,abs=1e-7)
    check_length(ref.length)
    for mutant in (lambda parts:sum(b-a for a,b in parts), lambda parts:int(ref.length(parts))):
        with pytest.raises(AssertionError):
            check_length(mutant)
    with pytest.raises(AssertionError):
        # A mutant that combines parts wrongly reports a section here.
        mutant=lambda *args,**kwargs: [{'common_length':300}]
        assert mutant()==ref.pair_coverage([[(0,0),(0,150)],[(0,400),(0,550)]],
                                          [[(2,0),(2,150)],[(2,400),(2,550)]])


def test_unique_union_across_repeated_qualifying_sections():
    sections=ref.pair_coverage([[(0,0),(0,300)]], [[(2,0),(2,300)]])
    assert ref.unique_pair_length(sections+sections)==pytest.approx(300,abs=1e-7)


def test_reference_is_independent_and_bounded():
    tree=ast.parse(Path(ref.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert all(not (name or '').startswith('pipeline_calculator') for name in imports)
    with pytest.raises(ValueError,match='six'):
        ref.common_axis_savings([(0,0,300)]*7)
