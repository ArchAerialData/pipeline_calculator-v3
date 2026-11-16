"""Shared local GIS import API, independent of desktop UI."""
from .models import ImportDecisionRequired, ImportFailure, VectorParseResult


def prepare_import(path, *, context=None):
    from .session import prepare_import as prepare
    return prepare(path, context=context)


__all__ = ['prepare_import', 'ImportDecisionRequired', 'ImportFailure', 'VectorParseResult']
