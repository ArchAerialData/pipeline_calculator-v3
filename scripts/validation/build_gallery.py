"""Generate source/corridor overlays and independent numerical geometry evidence."""
from pathlib import Path
import argparse
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.validation.common import environment, gallery_specs, geographic, save_json, write_fixture
from scripts.validation.geometry import inside, local_points, ring_checks, segment_distance
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.export.corridor_kml import build_overlap_corridor_kml

NS={'k':'http://www.opengis.net/kml/2.2'}


def run(output):
    rows=[]
    for spec in gallery_specs():
        origin=spec.get('origin',(-100,40));bearing=spec.get('bearing',0)
        path=write_fixture(output/f"{spec['id']}-source.kml",spec['paths'],origin=origin,bearing=bearing)
        result=PipelineAnalyzer(min_parallel_length=spec['minimum']).analyze_complete(path)
        sections=(result.get('overlap_analysis') or {}).get('bundled_sections',[])
        if not sections:
            rows.append({'fixture':spec['id'],'status':'no-sections','viewer':'pending'})
        for index,section in enumerate(sections,1):
            row={'fixture':spec['id'],'section':index,'viewer':'pending','status':'validated-approximation'}
            rows.append(row)
            row['raw']=ring_checks(local_points(section['corridor_polygon'],origin))
            try:
                document=ET.fromstring(build_overlap_corridor_kml(section,index))
            except ValueError as exc:
                row.update(status='unavailable',reason=str(exc));continue
            text=document.find('.//k:Polygon//k:coordinates',NS).text
            coords=[tuple(map(float,item.split(',')[:2])) for item in text.split()]
            xy=local_points(coords,origin)
            row['serialized']=ring_checks(xy)
            row['range_valid']=all(-180<=lon<=180 and -90<=lat<=90 for lon,lat in coords)
            row['description']=document.find('.//k:description',NS).text
            row['max_source_path_endpoint_outside_m']=0.0
            for pipeline,path_index in zip((section['pipeline_1'],section['pipeline_2']),section['source_path_indices']):
                part=spec['paths'][int(pipeline)][path_index]
                for point in (part[0],part[-1]):
                    p=local_points([geographic(point,origin,bearing)],origin)[0]
                    distance=0 if inside(p,xy) else min(segment_distance(p,a,b) for a,b in zip(xy,xy[1:]))
                    row['max_source_path_endpoint_outside_m']=max(row['max_source_path_endpoint_outside_m'],distance)
            if not row['serialized']['closed'] or row['serialized']['self_intersections'] or not row['range_valid'] or row['serialized']['area_m2']<=0:
                row['status']='failed'
            source=ET.parse(path).getroot().find('k:Document',NS)
            target=document.find('k:Document',NS)
            for placemark in source:
                target.append(placemark)
            ET.ElementTree(document).write(output/f"{spec['id']}-{index:02d}-overlay.kml",encoding='utf-8',xml_declaration=True)
    save_json(output/'report.json',{'environment':environment(),'cases':rows})
    (output/'report.md').write_text('# Corridor gallery\n\nSource-path endpoint shortfalls are informational: '
        'a qualified sampled section may cover only part of that source path. All viewer checks remain pending.\n\n'
        '| Fixture | Section | Raw crossings | Export status | Source endpoint outside m |\n| --- | --- | --- | --- | --- |\n'+
        '\n'.join(f"| {r['fixture']} | {r.get('section','')} | {r.get('raw',{}).get('self_intersections','')} | {r['status']} | {r.get('max_source_path_endpoint_outside_m','')} |" for r in rows)+'\n',encoding='utf-8')
    print(f'{len(rows)} gallery rows; failures: {sum(r["status"]=="failed" for r in rows)}')
    return int(any(r['status']=='failed' for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raise SystemExit(run(args.output))
