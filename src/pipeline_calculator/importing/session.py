"""Immutable captured sources with explicit layer/CRS decisions and fresh parses."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading

from .archive import capture, staged
from .coordinates import parse_crs
from .limits import MAX_LAYERS, check, report
from .models import ImportDecisionRequired, ImportFailure
from .normalize import Normalizer
from .worker import run_native


class VectorSourceSession:
    requires_repair = False
    can_save = False
    approved = True
    save_unavailable_reason = 'GIS inputs are retained unchanged; repaired-copy export applies only to KML/KMZ.'

    def __init__(self, snapshot, layers, versions, *, selected_layers=None, crs_overrides=None):
        self._snapshot = snapshot
        self._layers = deepcopy(layers)
        self._versions = dict(versions)
        eligible = [item['id'] for item in layers if item['supported']]
        self._selected_layers = list(selected_layers) if selected_layers is not None else eligible if len(eligible) == 1 else []
        self._crs_overrides = dict(crs_overrides or {})
        self._baseline = None
        self._lock = threading.RLock()
        self.original_path = snapshot.original_path
        self.display_name = Path(snapshot.original_path).name
        self.effective_format = Path(snapshot.original_path).suffix.lower().lstrip('.')
        self.token = hashlib.sha256((snapshot.digest + json.dumps([self._selected_layers, self._crs_overrides], sort_keys=True)).encode()).hexdigest()

    @property
    def layers(self):
        return [{key: deepcopy(value) for key, value in layer.items() if not key.startswith('_')}
                for layer in self._layers]

    @property
    def selected_layers(self):
        return list(self._selected_layers)

    @property
    def crs_overrides(self):
        return dict(self._crs_overrides)

    @property
    def verified(self):
        with self._lock:
            return self._baseline is not None

    @property
    def report(self):
        return {'status': 'ready' if self.verified else 'pending', 'original_name': self.display_name,
                'effective_format': self.effective_format, 'rules': [], 'source_sha256': self._snapshot.digest}

    @contextmanager
    def lease(self):
        # Staging is scoped to active native operations, never retained in results.
        # Worker references keep immutable captured bytes alive across owner close.
        yield self

    def retain(self):
        return self

    def release(self):
        self.close()

    def close(self):
        """Release owner use; Python references retain snapshots for active workers."""

    def with_selection(self, selected_layers, crs_overrides=None, *, context=None):
        check(context)
        if isinstance(selected_layers, (str, bytes)):
            raise ImportFailure('Select layer IDs as a list.', code='invalid_layer_selection')
        selection = list(selected_layers or [])
        known = {item['id']: item for item in self._layers}
        if (not selection or any(not isinstance(item, str) for item in selection)
                or len(selection) != len(set(selection)) or any(item not in known or not known[item]['supported'] for item in selection)):
            raise ImportFailure('Select at least one available vector layer, without duplicate selections.', code='invalid_layer_selection')
        assignments = dict(self._crs_overrides)
        for layer_id, value in (crs_overrides or {}).items():
            if layer_id not in selection or parse_crs(value) is None:
                raise ImportFailure('Assign a valid EPSG code or WKT to a selected layer.', code='invalid_crs')
            assignments[layer_id] = parse_crs(value).to_wkt()
        return VectorSourceSession(self._snapshot, self._layers, self._versions,
                                   selected_layers=selection, crs_overrides=assignments)

    def fresh_parse(self, *, context=None):
        check(context)
        if not self._selected_layers:
            raise ImportDecisionRequired(self, 'Choose the vector layers to analyze.', selection_required=True)
        selected = [next(item for item in self._layers if item['id'] == identifier) for identifier in self._selected_layers]
        missing = [item['id'] for item in selected if parse_crs(self._crs_overrides.get(item['id'], item['crs'])) is None]
        if missing:
            raise ImportDecisionRequired(self, 'Specify the source coordinate system for the selected layers.', missing_crs=missing)
        with self._lock:
            if self._baseline is not None:
                return deepcopy(self._baseline)
        normalizer = Normalizer(self._snapshot.digest, context)
        unsupported_packages = [member for member in self._snapshot.ignored
                                if Path(member).suffix.lower() in ('.kml', '.kmz', '.zip')]
        if unsupported_packages:
            normalizer.diagnostic('unsupported_package_members',
                'This package also contains KML documents or nested archives that were not analyzed. '
                'Open KML/KMZ inputs separately; nested ZIP/KMZ packages are not opened automatically.',
                level='warning', member_count=len(unsupported_packages), members=unsupported_packages)
        for descriptor in selected:
            check(context)
            dataset = next(item for item in self._snapshot.datasets if item['id'] == descriptor['dataset'])
            report(context, 'Reading selected GIS layer')
            with staged(self._snapshot, dataset) as path:
                packets = run_native(path, dataset['driver'], 'read', descriptor['_reader_name'], context)
                try:
                    normalizer.add_layer(descriptor, packets,
                        self._crs_overrides.get(descriptor['id'], descriptor['crs']),
                        assignment=self._crs_overrides.get(descriptor['id']))
                finally:
                    packets.close()
        parsed = normalizer.finish()
        parsed.import_metadata.update({'source_filename': self.display_name, 'reader_versions': self._versions,
                                       'ignored_members': list(self._snapshot.ignored),
                                       'discovered_layers': [{key: value for key, value in item.items() if not key.startswith('_')} for item in self._layers]})
        check(context)
        with self._lock:
            if self._baseline is None:
                self._baseline = parsed
            return deepcopy(self._baseline)


def prepare_import(path, *, context=None):
    suffix = Path(path).suffix.lower()
    if suffix in ('.kml', '.kmz'):
        from pipeline_calculator.parsers.source import prepare_source
        return prepare_source(path, context=context)
    snapshot = capture(path, context)
    if snapshot.kml_package:
        from pipeline_calculator.parsers.source import prepare_source
        with tempfile.TemporaryDirectory(prefix='pipeline-kml-') as directory:
            staged_path = Path(directory) / (Path(path).stem + '.kmz')
            staged_path.write_bytes(snapshot.archive_bytes)
            source = prepare_source(staged_path, context=context)
        # A KMZ snapshot owns all its bytes; source feature identities are member
        # names, so no temporary pathname escapes into analysis records.
        source.original_path = snapshot.original_path
        source.display_name = Path(path).name
        report_value = source.report
        report_value.update(original_name=source.display_name, source_filename=source.display_name,
                            supplied_extension='.zip')
        source._report = json.dumps(report_value, ensure_ascii=False, allow_nan=False)
        return source
    if not snapshot.datasets:
        raise ImportFailure('No supported GIS datasets were found in this package.', code='no_supported_features')
    layers, versions = [], {}
    for dataset in snapshot.datasets:
        report(context, 'Discovering GIS layers')
        with staged(snapshot, dataset) as staged_path:
            for packet in run_native(staged_path, dataset['driver'], 'discover', context=context):
                versions.update(packet['versions'])
                for descriptor in packet['layers']:
                    descriptor['_reader_name'] = descriptor['name']
                    # Single-file formats derive GDAL layer names from staged
                    # names; expose the original dataset name instead.
                    if dataset['format'] != 'gpkg':
                        descriptor['name'] = Path(dataset['id']).stem
                    descriptor.update(id=dataset['id'] + '::' + descriptor['name'],
                                      dataset=dataset['id'], format=dataset['format'])
                    layers.append(descriptor)
                    if len(layers) > MAX_LAYERS:
                        raise ImportFailure('Package exceeds the layer budget.', code='layer_limit')
    if not any(item['supported'] for item in layers):
        raise ImportFailure('No supported line or point layers are available. Polygon/raster/nonspatial layers do not contribute pipelines or pins.', code='no_supported_features')
    return VectorSourceSession(snapshot, layers, versions)
