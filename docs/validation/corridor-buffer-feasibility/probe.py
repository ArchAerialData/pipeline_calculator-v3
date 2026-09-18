import sys, math, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Polygon
from shapely.ops import transform, unary_union
from pyproj import Transformer
sys.path.insert(0, str(Path('src').resolve()))
sys.path.insert(0, r'C:\Users\rbake\Desktop\VS Code Shortcuts\Create-Polygon-Border-From-JPGs\src')
from pipeline_calculator.core.analyzer import PipelineAnalyzer
from pipeline_calculator.core.bundling import qualifying_sections
from pipeline_calculator.core.corridor_coverage import MeasuredPath
from pipeline_calculator.core.corridor_geometry import prepare_corridor
from polyline_boundary import build_polyline_boundary
out=Path('.validation-output/corridor-reuse')
out.mkdir(parents=True, exist_ok=True)
image=Image.new('RGB',(1120,820),'#17212b');draw=ImageDraw.Draw(image)
font=lambda size:ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',size)
draw.text((28,16),'Corridor shape: current output vs. reference buffer',(235,240,246),font=font(25))
draw.text((28,58),'Same qualified paths. White lines show the pipeline geometry.',(180,194,210),font=font(17))
rows=[]
for row,(name,xy) in enumerate([('L bend',[(0,0),(0,300),(300,300)]),('U bend',[(0,0),(0,300),(300,300),(300,0)])]):
    analyzer=PipelineAnalyzer(segment_length=10,min_parallel_length=50)
    def lonlat(x,y):return analyzer.geod.fwd(-100,40,math.degrees(math.atan2(x,y)),math.hypot(x,y))[:2]
    paths=[[lonlat(x+shift,y) for x,y in xy] for shift in (0,8)]
    pipes=[{'name':str(i),'coordinates':path} for i,path in enumerate(paths)]
    matches=analyzer.find_parallel_segments(pipes)
    results=analyzer.calculate_overlap_results(pipes,matches)
    qualified=qualifying_sections(pipes,matches,10,50)
    assert len(qualified)==len(results['bundled_sections'])==1
    q=qualified[0];spans=[]
    for index,path_index,ids in zip(q['pair'],q['paths'],q['segment_ids']):
        localids=[pipes[index]['segments'][i]['path_segment_index'] for i in ids]
        spans.append(MeasuredPath(analyzer.geod,paths[index]).span(min(localids)*10,(max(localids)+1)*10))
    prepared=prepare_corridor(results['bundled_sections'][0])
    reference=build_polyline_boundary([LineString(p) for p in spans],5/0.3048)
    projection=Transformer.from_crs(4326,reference.epsg,always_xy=True)
    current=unary_union([transform(projection.transform,Polygon(p['outer'],p['holes'])) for p in prepared['visualization_polygons']])
    source=[transform(projection.transform,LineString(p)) for p in spans]
    candidate=reference.final_polygon_projected
    assert candidate.is_valid and candidate.covers(unary_union(source))
    bounds=unary_union([current,candidate]).bounds
    xmin,ymin,xmax,ymax=bounds
    scale=min(440/(xmax-xmin),240/(ymax-ymin))
    for col,shape in enumerate((current,candidate)):
        left=28+col*548;top=104+row*322
        draw.rounded_rectangle((left,top,left+520,top+304),radius=12,fill='#22313f')
        draw.text((left+15,top+10),name+' - '+('current rectangle' if col==0 else 'reference buffer'),fill='#ecf0f4',font=font(19))
        cx=left+260;cy=top+170
        def screen(point):
            x,y=point;return (cx+(x-(xmin+xmax)/2)*scale,cy-(y-(ymin+ymax)/2)*scale)
        for poly in ([shape] if shape.geom_type=='Polygon' else shape.geoms):
            draw.polygon([screen(p) for p in poly.exterior.coords],fill='#675333' if col==0 else '#236d80',outline='#e0b77b' if col==0 else '#69d1dd')
            for hole in poly.interiors:draw.polygon([screen(p) for p in hole.coords],fill='#22313f')
        for line in source:draw.line([screen(p) for p in line.coords],fill='#ffffff',width=2)
    rows.append({'case':name,'qualified_length_m':q['length'],'current_kind':prepared['visualization_kind'],'current_area_m2':current.area,'reference_area_m2':candidate.area,'reference_valid':candidate.is_valid,'reference_covers_qualified_paths':candidate.covers(unary_union(source))})
draw.text((28,760),'Feasibility probe only: 5 m outward buffer; current detection range 15 m, sample step 10 m.',fill='#b4c2d2',font=font(16))
draw.text((28,786),'Production behavior is unchanged. State-border and complex-network validation is still required.',fill='#b4c2d2',font=font(16))
image.save(out/'comparison.png')
(out/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
print(json.dumps(rows,indent=2))
