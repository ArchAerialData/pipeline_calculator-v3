"""State analysis orchestration and accounting; geometry lives in geography.

All reported original mileage comes from the canonical source-interval ledger.
Overlap is calculated solely on exclusive interior paths. Shared-border lengths
are accounting allocations, never duplicated pipeline inputs to the analyzer.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math

from pipeline_calculator.core.coordinates import coordinate_paths_for_pipeline
from pipeline_calculator.core.corridor_coverage import MeasuredPath
from pipeline_calculator.core.execution import AnalysisCancelled, ScopedExecutionContext


ACCOUNTING_POLICY = (
    'Shared-border mileage is allocated equally to adjoining states at every length. '
    'Shared allocations receive no state overlap discount. State overlap uses exclusive '
    'interior geometry; state adjusted totals can differ from combined analysis.'
)


def _scope(context, start, end, label=''):
    return ScopedExecutionContext(context, start, end, label) if context is not None else None


def _diagnostic(code, message, *, level='error', **details):
    return {'level': level, 'code': code, 'message': message, 'context': details}


def _source_table(pipelines):
    # The parser assigns these IDs independently of names, OBJECTIDs and XML IDs.
    sources = {int(p.get('id', index)): p for index, p in enumerate(pipelines)}
    if len(sources) != len(pipelines):
        raise ValueError('Source pipeline identities must be unique')
    return sources


def _unresolved_fragments(pipelines, geod, context):
    """Retain source lines for a clearly incomplete export when resources fail."""
    fragments = []
    for source_id, pipeline in _source_table(pipelines).items():
        for path_index, coords in enumerate(coordinate_paths_for_pipeline(pipeline, context=context)):
            path = MeasuredPath(geod, coords, context=context)
            length = path.chainage[-1]
            if length <= 0:
                continue
            fragments.append({
                'id': f'{source_id}:{path_index}:unresolved', 'source_id': source_id,
                'source_name': pipeline.get('name', ''), 'placemark_id': pipeline.get('placemark_id', 'N/A'),
                'objectid': pipeline.get('objectid', 'N/A'), 'source_kml': pipeline.get('source_kml', ''),
                'path_index': path_index, 'start_m': 0.0, 'end_m': length,
                'length_meters': length, 'coordinates': [list(p) for p in coords],
                'kind': 'unresolved', 'state_codes': [],
            })
    return fragments


def _reconcile(fragments, total, attributed=None):
    sums = {kind: math.fsum(f['length_meters'] for f in fragments if f['kind'] == kind)
            for kind in ('state', 'shared', 'outside', 'unresolved')}
    accounted = math.fsum(sums.values())
    tolerance = max(.001, total * 1e-10)
    assigned = sums['state'] + sums['shared'] if attributed is None else attributed
    delta = math.fsum((assigned, sums['outside'], sums['unresolved'], -total))
    return {
        'source_meters': total, 'interior_meters': sums['state'], 'shared_meters': sums['shared'],
        'outside_meters': sums['outside'], 'unresolved_meters': sums['unresolved'],
        'attributed_state_meters': assigned, 'delta_meters': delta, 'tolerance_meters': tolerance,
        'passed': abs(accounted-total) <= tolerance and abs(delta) <= tolerance,
    }


def _state_inputs(fragments, sources, code):
    paths = defaultdict(list)
    for fragment in fragments:
        if fragment['kind'] == 'state' and fragment['state_codes'] == [code]:
            paths[fragment['source_id']].append(fragment['coordinates'])
    result = []
    for source_id in sorted(paths):
        pipeline = dict(sources[source_id])
        pipeline.update(id=source_id, coordinates=paths[source_id][0], coordinate_paths=paths[source_id])
        result.append(pipeline)
    return result


def _state_rows(fragments, sources, code, survey_mile):
    interior, shared, references = defaultdict(list), defaultdict(list), defaultdict(list)
    for fragment in fragments:
        if code not in fragment['state_codes']:
            continue
        source_id = fragment['source_id']
        if fragment['kind'] == 'state':
            interior[source_id].append(fragment['length_meters'])
        elif fragment['kind'] == 'shared':
            allocation = fragment['length_meters'] / len(fragment['state_codes'])
            shared[source_id].append(allocation)
            references[source_id].append({'fragment_id': fragment['id'], 'allocated_meters': allocation})
    rows = []
    for source_id in sorted(set(interior) | set(shared)):
        source = sources[source_id]
        measured, allocated = math.fsum(interior[source_id]), math.fsum(shared[source_id])
        total = measured + allocated
        rows.append({
            'source_id': source_id, 'Placemark_ID': source.get('placemark_id') or 'N/A',
            'OBJECTID': source.get('objectid', 'N/A'), 'Name': source.get('name', ''),
            'Shape_Length': total, 'pipelinelength': total / survey_mile,
            'interior_meters': measured, 'shared_allocation_meters': allocated,
            'shared_border_allocations': references[source_id],
        })
    return rows


def build_state_breakdown(analyzer, pipelines, combined, *, context=None, boundaries=None):
    """Build all state scopes, retaining combined results on optional failures."""
    total = combined['total_meters']
    geography = {
        'schema_version': 1, 'status': 'unavailable', 'analysis_complete': False,
        'boundary_source': {}, 'accounting_policy': ACCOUNTING_POLICY,
        'states': [], 'fragments': [], 'diagnostics': [],
    }
    try:
        from pipeline_calculator.core.geography import load_boundaries, partition_pipelines, clip_state_corridor
        loading = _scope(context, .43, .46)
        if loading is not None:
            loading.report('Loading state boundaries', 0, 1)
        if boundaries is None:
            boundaries = load_boundaries()
        geography['boundary_source'] = dict(boundaries.boundary_source)
        if loading is not None:
            loading.report('Loading state boundaries', 1, 1)
        partition = partition_pipelines(pipelines, analyzer.geod,
                                       context=_scope(context, .46, .66), boundaries=boundaries)
        geography['fragments'] = fragments = partition['fragments']
        geography['crossing_count'] = partition.get('crossing_count', 0)
        geography['partition_audit'] = partition.get('reconciliation', {})
        geography['diagnostics'].extend(partition.get('diagnostics', []))
        names = partition.get('state_names', boundaries.state_names)
        sources = _source_table(pipelines)
        codes = sorted({code for f in fragments if f['kind'] in ('state', 'shared')
                        for code in f['state_codes']}, key=lambda c: (names[c], c))
        input_diagnostics = [d for d in combined.get('diagnostics', [])
                             if d.get('code') != 'overlap_analysis_failed']
        # Do not infer single-state identity merely from the bounding box.
        single_state = len(codes) == 1 and all(
            f['kind'] == 'state' and f['state_codes'] == codes for f in fragments)
        for index, code in enumerate(codes):
            state_context = _scope(context, .66 + .32*index/len(codes),
                                   .66 + .32*(index+1)/len(codes),
                                   f'Analyzing {names[code]} ({index+1} of {len(codes)})')
            if state_context is not None:
                state_context.check()
            inputs = _state_inputs(fragments, sources, code)
            rows = _state_rows(fragments, sources, code, analyzer.survey_mile)
            interior = math.fsum(row['interior_meters'] for row in rows)
            allocated = math.fsum(row['shared_allocation_meters'] for row in rows)
            try:
                state = (deepcopy(combined) if single_state else analyzer.analyze_features(
                    inputs, diagnostics=input_diagnostics, parsed_kml_files=combined.get('parsed_kml_files'),
                    context=state_context))
                if abs(state['total_meters']-interior) > max(.001, interior*1e-10):
                    raise ValueError('Clipped geometry length does not agree with its source intervals')
            except AnalysisCancelled:
                raise
            except Exception as exc:
                state = {'overlap_analysis': None, 'analysis_complete': False,
                         'diagnostics': list(input_diagnostics) + [_diagnostic(
                             'state_analysis_failed', 'State overlap calculation is unavailable.',
                             state=code, error=str(exc))],
                         'analysis_parameters': dict(combined['analysis_parameters']),
                         'parsed_kml_files': list(combined.get('parsed_kml_files', []))}
            overlap = state.get('overlap_analysis')
            overlap_failed = any(d.get('code') in ('overlap_analysis_failed', 'state_analysis_failed')
                                 for d in state.get('diagnostics', []))
            savings = None if overlap_failed else (overlap['savings_meters'] if overlap else 0.0)
            attributed = interior + allocated
            adjusted = None if savings is None else max(0.0, attributed-savings)
            if savings is not None:
                if overlap is None:
                    overlap = {'bundled_sections': [], 'pipeline_overlaps': {}, 'total_bundled_length': 0.0}
                overlap.update(savings_meters=savings, savings_miles=savings/analyzer.survey_mile,
                               savings_percentage=savings/attributed*100 if attributed else 0.0,
                               effective_total_meters=adjusted, effective_total_miles=adjusted/analyzer.survey_mile,
                               computation_method='state_interior_qualified_segment_coverage_v1')
                clipped_sections = []
                for section_index, section in enumerate(overlap.get('bundled_sections', [])):
                    if state_context is not None:
                        state_context.checkpoint()
                    try:
                        clipped = clip_state_corridor(section, code, boundaries, analyzer.geod,
                                                      context=state_context)
                    except AnalysisCancelled:
                        raise
                    except Exception as exc:
                        clipped = dict(section, clipped_polygons=[], diagnostics=[_diagnostic(
                            'state_corridor_unavailable', 'A state corridor visualization was omitted.',
                            level='warning', state=code, section=section_index, error=str(exc))])
                    clipped['state_code'], clipped['state_name'] = code, names[code]
                    state['diagnostics'].extend(clipped.get('diagnostics', []))
                    clipped_sections.append(clipped)
                overlap['bundled_sections'] = clipped_sections
            state.update(
                state_code=code, state_name=names[code], pipelines=rows, placemarks=[],
                total_meters=attributed, total_miles=attributed/analyzer.survey_mile,
                interior_meters=interior, shared_allocation_meters=allocated,
                interior_savings_meters=savings, adjusted_total_meters=adjusted,
                overlap_analysis=overlap, shared_overlap_status='not_calculated' if allocated else 'not_applicable',
                accounting_policy=ACCOUNTING_POLICY,
            )
            for pipeline in inputs:
                pipeline.pop('segments', None)
            geography['states'].append(state)
            if state_context is not None:
                state_context.finish()
        geography['reconciliation'] = _reconcile(
            fragments, total, math.fsum(s['total_meters'] for s in geography['states']))
        if partition.get('reconciliation', {}).get('passed') is not True:
            geography['reconciliation']['passed'] = False
        # Check each source independently too; opposing mistakes cannot cancel.
        source_partition_lengths = defaultdict(list)
        for fragment in fragments:
            source_partition_lengths[fragment['source_id']].append(fragment['length_meters'])
        for source_id, source in sources.items():
            expected = math.fsum(MeasuredPath(analyzer.geod, coords, context=context).chainage[-1]
                                 for coords in coordinate_paths_for_pipeline(source, context=context))
            actual = math.fsum(source_partition_lengths[source_id])
            if abs(actual-expected) > max(.001, expected*1e-10):
                geography['diagnostics'].append(_diagnostic(
                    'source_partition_mismatch', 'State partition did not preserve a source pipeline length.',
                    source_id=source_id, expected_meters=expected, actual_meters=actual))
                geography['reconciliation']['passed'] = False
        if not geography['reconciliation']['passed']:
            geography['diagnostics'].append(_diagnostic(
                'state_reconciliation_failed', 'State mileage does not reconcile; do not use this breakdown as complete.'))
        if geography['reconciliation']['unresolved_meters'] > 0:
            geography['diagnostics'].append(_diagnostic(
                'state_geometry_unresolved', 'Some geometry could not be assigned to a state. Its mileage is retained separately.'))
        partition_incomplete = (geography['reconciliation']['unresolved_meters'] > 0
                                or not geography['reconciliation']['passed'])
        if partition_incomplete:
            for state in geography['states']:
                state['analysis_complete'] = False
                state['diagnostics'].append(_diagnostic(
                    'state_partition_incomplete',
                    'This state result covers reliably assigned geometry only; the input partition is incomplete.'))
        if geography['reconciliation']['outside_meters'] > 0:
            geography['diagnostics'].append(_diagnostic(
                'outside_state_coverage', 'Some mileage lies outside the 50 states and Washington, DC.', level='warning'))
        complete = (not any(d.get('level') == 'error' for d in input_diagnostics + geography['diagnostics'])
                    and all(s['analysis_complete'] for s in geography['states']))
        geography.update(analysis_complete=complete, status='complete' if complete else 'incomplete')
    except AnalysisCancelled:
        raise
    except Exception as exc:
        geography.update(states=[], status='unavailable', analysis_complete=False)
        geography['diagnostics'].append(_diagnostic(
            'state_breakdown_unavailable', 'State breakdown is unavailable; combined results remain available.', error=str(exc)))
        geography['fragments'] = _unresolved_fragments(pipelines, analyzer.geod, context)
        geography['reconciliation'] = _reconcile(geography['fragments'], total, 0.0)
    if context is not None:
        context.report('Finalizing state results', 1, 1, fraction=.99)
    return geography
