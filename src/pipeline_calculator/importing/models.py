"""Format-neutral results and explicit, UI-independent import decisions."""
from dataclasses import dataclass, field
from pipeline_calculator.parsers.kml_kmz import ParseResult


class ImportFailure(ValueError):
    def __init__(self, message, *, code='import_failed', findings=None):
        super().__init__(message)
        self.code = code
        self.findings = findings if findings is not None else [{
            'code': code, 'category': 'import', 'message': message,
        }]


class ImportDecisionRequired(ValueError):
    def __init__(self, source_session, message, *, missing_crs=None, selection_required=False):
        super().__init__(message)
        self.source_session = source_session
        self.missing_crs = list(missing_crs or [])
        self.selection_required = bool(selection_required)


@dataclass
class VectorParseResult(ParseResult):
    parsed_sources: list[dict] = field(default_factory=list)
    import_metadata: dict = field(default_factory=dict)
