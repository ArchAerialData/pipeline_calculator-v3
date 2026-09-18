"""Cheap display decisions from completed corridor results; no topology work."""
from pipeline_calculator.core.corridor_geometry import has_canonical_geometry


def corridor_is_omitted(section):
    if not has_canonical_geometry(section):
        return False  # Legacy sections can construct their documented approximation.
    if 'visualization_status' in section and section['visualization_status'] != 'ready':
        return True
    if ('visualization_schema_version' in section and
            (type(section['visualization_schema_version']) is not int or
             section['visualization_schema_version'] != 1 or
             section.get('visualization_status') != 'ready')):
        return True
    # Clipped geometry is authoritative; never substitute an uncut fallback.
    polygons = section.get('clipped_polygons', section.get('visualization_polygons'))
    return type(polygons) is not list or not polygons


_REASONS = {
    'corridor_buffer_limit': 'The map exceeded a drawing complexity or resource limit.',
    'corridor_projection_unavailable': 'The map could not meet its geographic accuracy checks.',
    'corridor_geometry_invalid': 'The map shape could not be safely constructed.',
    'corridor_coverage_failed': 'The map could not reliably cover all qualifying pipeline paths.',
    'state_corridor_omitted': 'The map could not be safely clipped and verified within this state.',
    'state_corridor_unavailable': 'The state corridor map could not be prepared.',
}


def corridor_unavailable_reason(section):
    """Explain known structured failures without interpreting arbitrary error text."""
    if not corridor_is_omitted(section):
        return ''
    version = section.get('visualization_schema_version', 1)
    if type(version) is not int or version != 1:
        return "This map's saved format is not supported by this app version."
    codes = []
    for diagnostic in section.get('diagnostics') or []:
        if not isinstance(diagnostic, dict):
            continue
        context = diagnostic.get('context')
        if isinstance(context, dict):
            codes.append(context.get('reason_code'))
        codes.append(diagnostic.get('code'))
    # Preserve a specific construction failure when state clipping also reports
    # a generic omission of the already-unavailable map.
    for code in _REASONS:
        if code in codes:
            return _REASONS[code]
    if section.get('visualization_status') == 'ready':
        return "The map's polygon data is missing or incomplete."
    if section.get('visualization_status') not in (None, 'omitted'):
        return "The map's result status is not supported."
    return 'The corridor map could not be generated.'


MAP_OMISSION_NOTE = (
    'This map omission does not change calculated mileage. See Diagnostics for technical details.'
)
