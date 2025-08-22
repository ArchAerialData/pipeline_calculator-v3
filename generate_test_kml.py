"""Utility to generate sample KML files for testing the Pipeline Calculator."""
from __future__ import annotations
import argparse
import os

def generate_test_kml(output_path: str = "test_data/sample.kml") -> str:
    """Generate a simple KML file containing two pipeline segments.

    Args:
        output_path: Path where the KML file will be written.

    Returns:
        The path to the generated KML file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    kml_content = """<?xml version='1.0' encoding='UTF-8'?>
<kml xmlns='http://www.opengis.net/kml/2.2'>
  <Document>
    <Placemark>
      <name>Pipeline A</name>
      <LineString>
        <coordinates>
          -100.0,40.0,0 -101.0,41.0,0
        </coordinates>
      </LineString>
    </Placemark>
    <Placemark>
      <name>Pipeline B</name>
      <LineString>
        <coordinates>
          -100.0,40.5,0 -101.0,41.5,0
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(kml_content)
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple KML file for testing.")
    parser.add_argument("output", nargs="?", default="test_data/sample.kml", help="Output KML file path")
    args = parser.parse_args()
    path = generate_test_kml(args.output)
    print(f"Created {path}")

if __name__ == "__main__":
    main()
