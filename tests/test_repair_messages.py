"""User guidance must not turn detected problems into unverified success claims."""
import pytest

from pipeline_calculator.gui.repair_messages import failure_explanation, repair_explanation
from pipeline_calculator.parsers.repair import RepairFailure


def test_pending_and_verified_repairs_describe_different_outcomes():
    report = {'status': 'eligible', 'rules': ['missing_xsi_schema_namespace_v1']}
    before = repair_explanation(report)
    assert 'formatting label is missing' in before
    assert 'passed' not in before and 'was restored' not in before
    after = repair_explanation(dict(report, status='verified'))
    assert 'was restored' in after and 'verification passed' in after
    assert report['status'] == 'eligible'


def test_multiple_rules_are_concise_and_unknown_rules_are_not_guessed():
    rules = ['literal_metadata_ampersand_v1', 'leading_xml_whitespace_v1',
             'filename_format_mismatch_v1', 'future_rule']
    text = repair_explanation({'status': 'eligible', 'rules': rules * 50})
    assert '& character' in text and 'blank space' in text and 'file type' in text
    assert len(text) < 500
    unknown = repair_explanation({'status': 'eligible', 'rules': ['future_rule']})
    assert 'technical details' in unknown
    assert 'coordinates' not in unknown and 'passed' not in unknown


@pytest.mark.parametrize('category', ['verification', 'operation', 'limit'])
def test_application_failures_never_blame_client_geometry(category):
    error = RepairFailure('Internal issue', category=category, findings=[
        {'code': 'invalid_coordinate', 'message': 'Bad coordinates.', 'category': 'source'}])
    text = failure_explanation(error, client_request=True)
    assert 'support' in text
    assert 'Ask the client' not in text and 'verification passed' not in text


def test_incomplete_geometry_guidance_retains_exact_client_request_separately():
    error = RepairFailure('Analysis blocked', category='coverage', findings=[
        {'code': 'invalid_coordinate', 'message': 'A bad coordinate in pipeline A.', 'category': 'source'},
        {'code': 'future_code', 'message': 'An unsupported shape in pipeline B.', 'category': 'source'}])
    original = error.client_request()
    text = failure_explanation(error, client_request=True)
    assert 'missing or invalid' in text and 'pipeline B' in text
    assert 'corrected, complete export' in text
    assert error.client_request() == original
