"""Plain-language presentation of detected issues; never decides repair eligibility."""

_FORMATTING = {
    'missing_xsi_schema_namespace_v1': (
        'A required formatting label is missing. Repair can add it without changing pipeline coordinates.',
        'A missing formatting label was restored.'),
    'literal_metadata_ampersand_v1': (
        'An & character in descriptive text is incorrectly formatted. Repair can fix how that text is stored.',
        'The formatting of & characters in descriptive text was corrected.'),
    'leading_xml_whitespace_v1': (
        'Extra blank space appears before the file header. Repair can remove that space.',
        'Extra blank space before the file header was removed.'),
    'filename_format_mismatch_v1': (
        'The filename ending does not match the actual file type. The app can use the detected type.',
        'The app used the actual file type instead of the incorrect filename ending.'),
}

_SOURCE_ISSUES = {
    'invalid_coordinate': 'Some shape coordinates are missing or invalid.',
    'invalid_gx_coord': 'Some track coordinates are missing or invalid.',
    'missing_point_coordinate': 'A point is missing its location.',
    'malformed_xml': 'The file structure is still invalid after the supported formatting fixes.',
    'missing_geometry_namespace': 'A definition needed to interpret the shapes is missing.',
    'missing_kml_namespace': 'The file is missing the definition that identifies its KML shapes.',
    'unsupported_dtd': 'The file uses extra definitions that the app cannot safely verify.',
    'no_supported_centerlines': 'No supported pipeline centerlines were found.',
    'unsupported_repair_encoding': 'The app cannot safely determine how the file text is encoded.',
}


def repair_explanation(report):
    """Summarize known rules without promoting an eligible repair to a success."""
    verified = report.get('status') == 'verified'
    messages = []
    for rule in report.get('rules') or []:
        pair = _FORMATTING.get(rule)
        text = pair[int(verified)] if pair else 'Another formatting issue is listed in the technical details.'
        if text not in messages:
            messages.append(text)
    if not messages:
        return 'See the technical details for the recorded file checks.'
    if len(messages) > 3:
        messages = messages[:3] + ['Additional formatting changes are listed below.']
    if verified:
        messages.append('Geometry verification passed; the original file was preserved.')
    return ' '.join(messages)


def failure_explanation(error, *, client_request=False):
    """Keep application/verification limits separate from client-source problems."""
    category = getattr(error, 'category', '')
    if category == 'verification':
        return ('The app could not confirm that the repair preserves all geometry. '
                'Do not use it for mileage analysis; share the technical details with support.')
    if category == 'limit':
        return ('The file reached an application inspection limit. Contact support; '
                'do not remove pipelines to make the file fit.')
    if category == 'operation':
        return ('A file-access or processing step could not finish. Retry, or share the technical '
                'details with support if it continues. This does not establish that the shapes are damaged.')
    if not client_request:
        return ('The app could not safely prepare the complete file. Share the technical details '
                'with support before changing any geometry.')
    messages = []
    for finding in getattr(error, 'findings', []):
        if finding.get('category', 'source') not in ('source', 'policy'):
            continue
        text = _SOURCE_ISSUES.get(finding.get('code'))
        if text is None:
            # Preserve the detected cause for unfamiliar codes; never invent a repair.
            text = ' '.join(str(finding.get('message', '')).split())
            if len(text) > 180:
                text = text[:177].rsplit(' ', 1)[0] + '…'
        if text and text not in messages:
            messages.append(text)
    summary = ' '.join(messages[:2]) or 'The complete geometry could not be read safely.'
    if len(messages) > 2:
        summary += ' More issues are listed below.'
    return summary + ' Ask the client for a corrected, complete export. Use Copy client request for the specific issues.'
