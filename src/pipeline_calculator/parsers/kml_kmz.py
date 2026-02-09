from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET


def _extract_objectid(placemark, namespace):
    """Extract OBJECTID from a placemark, returning 'N/A' when unavailable."""
    try:
        objectid = "N/A"
        if namespace:
            objectid_elem = placemark.find('.//kml:Data[@name="OBJECTID"]/kml:value', namespace)
            if objectid_elem is None:
                objectid_elem = placemark.find('.//kml:SimpleData[@name="OBJECTID"]', namespace)
        else:
            objectid_elem = placemark.find('.//Data[@name="OBJECTID"]/value')
            if objectid_elem is None:
                objectid_elem = placemark.find('.//SimpleData[@name="OBJECTID"]')

        if objectid_elem is not None and objectid_elem.text:
            objectid = objectid_elem.text.strip()
        return objectid
    except Exception:
        return "N/A"


def _has_linestring(placemark, namespace):
    try:
        if namespace:
            return placemark.find(".//kml:LineString", namespace) is not None
        return placemark.find(".//LineString") is not None
    except Exception:
        return False


def _has_point(placemark, namespace):
    try:
        if namespace:
            return placemark.find(".//kml:Point", namespace) is not None
        return placemark.find(".//Point") is not None
    except Exception:
        return False


def _extract_coordinates(placemark, namespace):
    """Extract and parse coordinates from a placemark safely."""
    try:
        if namespace:
            coords_elem = placemark.find(".//kml:coordinates", namespace)
        else:
            coords_elem = placemark.find(".//coordinates")

        coords = []
        if coords_elem is not None and coords_elem.text:
            coords_text = coords_elem.text.strip()

            for coord_str in coords_text.replace("\n", " ").replace("\t", " ").split():
                coord_str = coord_str.strip()
                if not coord_str:
                    continue

                try:
                    parts = coord_str.split(",")
                    if len(parts) >= 2:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        # Validate coordinate ranges
                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                            coords.append((lon, lat))
                except (ValueError, IndexError):
                    continue
        return coords
    except Exception:
        return []


def extract_features_from_file(file_path, progress_callback=None):
    """Extract pipelines and point placemarks from a KMZ/KML file.

    Returns:
      pipelines: list[dict] with keys {id, objectid, name, coordinates}
      placemarks: list[dict] with keys {Placemark_ID, Name, Count}
    """

    def _open_kml(path):
        try:
            if path.lower().endswith(".kmz"):
                kmz = zipfile.ZipFile(path, "r")
                kml_files = [f for f in kmz.namelist() if f.lower().endswith(".kml")]
                if not kml_files:
                    raise ValueError("No KML file found in KMZ archive")
                return kmz, kmz.open(kml_files[0])
            return None, open(path, "rb")
        except Exception as e:
            raise ValueError(f"Failed to open file: {str(e)}")

    kmz, kml_file = _open_kml(file_path)

    pipelines = []
    placemark_data = []
    pipeline_count = 0
    placemark_count = 0

    try:
        try:
            context = ET.iterparse(kml_file, events=("start", "end"))
            _, root = next(context)
        except (StopIteration, ET.ParseError) as e:
            raise ValueError(f"Invalid or empty KML/KMZ file: {str(e)}")

        ns_match = re.match(r"\{(.*)\}", root.tag)
        ns = ns_match.group(1) if ns_match else ""
        namespace = {"kml": ns} if ns else None

        for event, elem in context:
            if event == "end" and elem.tag.endswith("Placemark"):
                try:
                    if namespace:
                        name_elem = elem.find("kml:name", namespace)
                    else:
                        name_elem = elem.find("name")

                    item_index = pipeline_count + placemark_count + 1
                    name = (
                        name_elem.text.strip()
                        if name_elem is not None and name_elem.text and name_elem.text.strip()
                        else f"Item_{item_index}"
                    )

                    objectid = _extract_objectid(elem, namespace)
                    coords = _extract_coordinates(elem, namespace)

                    if coords and len(coords) > 0:
                        has_linestring = _has_linestring(elem, namespace)
                        has_point = _has_point(elem, namespace)

                        if has_linestring or (len(coords) >= 2 and not has_point):
                            pipeline_count += 1
                            pipelines.append(
                                {
                                    "id": pipeline_count - 1,
                                    "objectid": objectid,
                                    "name": name,
                                    "coordinates": coords,
                                }
                            )
                        elif has_point or len(coords) == 1:
                            placemark_count += 1
                            placemark_data.append(
                                {
                                    "Placemark_ID": objectid
                                    if objectid != "N/A"
                                    else f"PM_{placemark_count}",
                                    "Name": name,
                                    "Count": 1,
                                }
                            )
                except Exception as e:
                    # Skip malformed placemarks but continue processing
                    print(f"Warning: Skipping malformed placemark: {str(e)}")
                    continue
                finally:
                    elem.clear()

    except Exception as e:
        raise ValueError(f"Error parsing KML data: {str(e)}")
    finally:
        try:
            kml_file.close()
            if kmz:
                kmz.close()
        except Exception:
            pass

    return pipelines, placemark_data


def parse_kml_kmz(file_path, progress_callback=None):
    """Alias for extract_features_from_file (preferred name for the refactor)."""
    return extract_features_from_file(file_path, progress_callback=progress_callback)

