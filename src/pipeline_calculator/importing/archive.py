"""Capture local inputs once; bounded ZIP inventory never extracts source paths."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import unicodedata
import zipfile

from .formats import DRIVERS
from .limits import CHUNK_BYTES, MAX_SOURCE_BYTES, MAX_EXPANDED_BYTES, MAX_ENTRIES, check, report
from .models import ImportFailure

SHP_COMPONENTS = {'.shp', '.shx', '.dbf', '.prj', '.cpg'}


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_stable(path, context=None):
    path = Path(path)
    check(context)
    try:
        with path.open('rb') as stream:
            before = _signature(os.fstat(stream.fileno()))
            if before[2] > MAX_SOURCE_BYTES:
                raise ImportFailure('Input exceeds the 256 MiB source limit.', code='source_size_limit')
            chunks, size = [], 0
            while True:
                check(context)
                chunk = stream.read(CHUNK_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_SOURCE_BYTES:
                    raise ImportFailure('Input exceeds the source size limit.', code='source_size_limit')
            after = _signature(os.fstat(stream.fileno()))
        if before != after or after != _signature(path.stat()):
            raise ImportFailure('The input changed while being captured. Close the exporter and retry.', code='source_changed')
        return b''.join(chunks), after
    except OSError as exc:
        raise ImportFailure(f'Cannot read {path.name}: {exc.strerror or type(exc).__name__}', code='source_unreadable') from exc


def safe_member(name):
    normalized = name.replace('\\', '/')
    parts = normalized.rstrip('/').split('/')
    if (not normalized or normalized.startswith('/') or '\x00' in normalized
            or any(part in ('', '.', '..') or ':' in part or part.endswith(('.', ' ')) for part in parts)):
        raise ImportFailure('ZIP contains an unsafe member path.', code='invalid_archive_path')
    return normalized.rstrip('/'), unicodedata.normalize('NFC', normalized.rstrip('/')).casefold()


@dataclass(frozen=True)
class CapturedInput:
    original_path: str
    digest: str
    members: tuple
    datasets: tuple
    ignored: tuple
    archive_bytes: bytes | None = None
    kml_package: bool = False

    @property
    def files(self):
        return dict(self.members)


def _dataset_inventory(files):
    by_portable = {unicodedata.normalize('NFC', name).casefold(): name for name in files}
    datasets = []
    for name in sorted(files):
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in DRIVERS:
            continue
        components = [name]
        if suffix == '.shp':
            stem = str(PurePosixPath(name).with_suffix(''))
            for extension in ('.shx', '.dbf', '.prj', '.cpg'):
                component = by_portable.get(unicodedata.normalize('NFC', stem + extension).casefold())
                if component:
                    components.append(component)
                elif extension in ('.shx', '.dbf'):
                    raise ImportFailure(f'{name} requires {stem + extension}. Supply the complete shapefile set.', code='missing_sidecar')
        if suffix == '.gpkg':
            for extension in ('-wal', '-journal', '-shm'):
                if unicodedata.normalize('NFC', name + extension).casefold() in by_portable:
                    raise ImportFailure(f'{name} has SQLite transaction sidecars. Export or close it before importing.', code='gpkg_live_transaction')
        datasets.append({'id': name, 'format': suffix.lstrip('.'), 'driver': DRIVERS[suffix],
                         'components': tuple(components)})
    return tuple(datasets)


def capture(path, context=None):
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if not path.is_file():
        raise ImportFailure('Choose one supported GIS file, not a directory.', code='unsupported_input')
    report(context, 'Capturing GIS input')
    if suffix == '.zip':
        raw, _ = read_stable(path, context)
        files, inventory, identities, total = {}, [], set(), 0
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                infos = archive.infolist()
                if len(infos) > MAX_ENTRIES:
                    raise ImportFailure('ZIP exceeds the archive entry limit.', code='archive_entry_limit')
                for index, info in enumerate(infos):
                    report(context, 'Inspecting ZIP package', index, len(infos))
                    name, identity = safe_member(info.filename)
                    if info.orig_filename != info.filename or identity in identities:
                        raise ImportFailure('ZIP has duplicate or ambiguous member names.', code='ambiguous_archive_path')
                    identities.add(identity)
                    if info.flag_bits & 1:
                        raise ImportFailure('Encrypted ZIP members are unsupported.', code='encrypted_archive')
                    mode = (info.external_attr >> 16) & 0xFFFF
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise ImportFailure('ZIP contains a link or special file.', code='special_archive_member')
                    if info.file_size < 0 or total + info.file_size > MAX_EXPANDED_BYTES:
                        raise ImportFailure('ZIP exceeds the 256 MiB expanded limit.', code='package_size_limit')
                    inventory.append(name)
                    keep = (PurePosixPath(name).suffix.lower() in set(DRIVERS) | SHP_COMPONENTS
                            or name.lower().endswith(('.gpkg-wal', '.gpkg-shm', '.gpkg-journal')))
                    chunks = []
                    with archive.open(info) as stream:
                        while True:
                            check(context)
                            chunk = stream.read(CHUNK_BYTES)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > MAX_EXPANDED_BYTES:
                                raise ImportFailure('ZIP exceeds the expanded byte limit.', code='package_size_limit')
                            if keep:
                                chunks.append(chunk)
                    if keep and not info.is_dir() and '__MACOSX' not in PurePosixPath(name).parts:
                        files[name] = b''.join(chunks)
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError, OSError) as exc:
            raise ImportFailure('ZIP is damaged, truncated, or uses unsupported compression.', code='invalid_archive') from exc
        # Generic packages commonly carry unrelated manifest/readme JSON files.
        # A loose .json still receives strict GeoJSON validation in the reader.
        for name in tuple(files):
            if PurePosixPath(name).suffix.lower() == '.json':
                try:
                    item = json.loads(files[name])
                    is_geojson = isinstance(item, dict) and item.get('type') in {
                        'FeatureCollection', 'Feature', 'Point', 'MultiPoint', 'LineString',
                        'MultiLineString', 'Polygon', 'MultiPolygon', 'GeometryCollection'}
                except (ValueError, UnicodeError, RecursionError):
                    is_geojson = False
                if not is_geojson:
                    del files[name]
        datasets = _dataset_inventory(files)
        kml_package = not datasets and any(name.lower().endswith('.kml') for name in inventory)
        used = {name for item in datasets for name in item['components']}
        ignored = tuple(name for name in inventory if name not in used)
        return CapturedInput(str(path), hashlib.sha256(raw).hexdigest(), tuple(files.items()), datasets,
                             ignored, raw, kml_package)
    if suffix not in DRIVERS:
        raise ImportFailure('Unsupported GIS input format.', code='unsupported_format')
    paths = [path]
    source_stem = unicodedata.normalize('NFC', path.stem).casefold()
    def shapefile_components():
        return [item for item in path.parent.iterdir() if item.is_file()
                and unicodedata.normalize('NFC', item.stem).casefold() == source_stem
                and item.suffix.lower() in SHP_COMPONENTS]
    if suffix == '.shp':
        siblings = shapefile_components()
        suffixes = [item.suffix.lower() for item in siblings]
        if len(set(suffixes)) != len(suffixes):
            raise ImportFailure('Shapefile has ambiguous case-variant sidecars.', code='ambiguous_sidecar')
        paths = siblings
    if suffix == '.gpkg' and any(Path(str(path) + extra).exists() for extra in ('-wal', '-shm', '-journal')):
        raise ImportFailure('GeoPackage has SQLite transaction sidecars. Close/export it before importing.', code='gpkg_live_transaction')
    files, signatures, total = {}, {}, 0
    for item in paths:
        data, signature = read_stable(item, context)
        total += len(data)
        if total > MAX_SOURCE_BYTES:
            raise ImportFailure('Shapefile components exceed the source byte limit.', code='source_size_limit')
        files[item.name], signatures[item] = data, signature
    for item, signature in signatures.items():
        if not item.exists() or _signature(item.stat()) != signature:
            raise ImportFailure('A shapefile component changed during capture.', code='source_changed')
    if suffix == '.shp' and set(shapefile_components()) != set(paths):
        raise ImportFailure('The shapefile component set changed during capture.', code='source_changed')
    if suffix == '.gpkg' and any(Path(str(path) + extra).exists() for extra in ('-wal', '-shm', '-journal')):
        raise ImportFailure('GeoPackage transaction state changed during capture.', code='source_changed')
    digest = hashlib.sha256()
    for name, data in sorted(files.items()):
        digest.update(name.encode('utf-8') + b'\x00' + hashlib.sha256(data).digest())
    return CapturedInput(str(path), digest.hexdigest(), tuple(files.items()), _dataset_inventory(files), ())


@contextmanager
def staged(snapshot, dataset):
    """Generated staging names; original member names never become destinations."""
    with tempfile.TemporaryDirectory(prefix='pipeline-gis-') as directory:
        files = snapshot.files
        output = None
        for member in dataset['components']:
            suffix = PurePosixPath(member).suffix.lower()
            destination = Path(directory) / ('dataset' + suffix)
            destination.write_bytes(files[member])
            if member == dataset['id']:
                output = destination
        yield output
