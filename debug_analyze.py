import sys
import os
sys.path.append('src')
from pipeline_calculator_v3 import PipelineAnalyzer

def main():
    an = PipelineAnalyzer()
    file = 'test_data/Brazos_NGL and Delaware_Gas combined.kmz'
    res = an.analyze_complete(file)
    print('pipelines:', len(res['pipelines']))
    print('total miles:', res['total_miles'])
    ov = res['overlap_analysis']
    if not ov:
        print('no overlap results')
        return
    print('bundled count:', len(ov['bundled_sections']))
    if ov['bundled_sections']:
        s = ov['bundled_sections'][0]
        print('first section miles:', s['bundled_length_miles'])
        print('avg sep:', s['average_separation'])
        print('oriented width m:', s.get('oriented_width_m'))
        poly = s.get('corridor_polygon') or s.get('oriented_polygon')
        print('poly points:', len(poly) if poly else None)
        if poly:
            for i, p in enumerate(poly[:20]):
                print(i, p)
            # Show cross distances between alternating points to see if ring alternates sides
            from pyproj import Geod
            geod = an.geod
            for i in range(0, 20, 2):
                lon1, lat1 = poly[i]
                lon2, lat2 = poly[i+1]
                _, _, d = geod.inv(lon1, lat1, lon2, lat2)
                print(f"{i}->{i+1}: {d:.2f} m")
            print('first 12 edge distances:')
            for i in range(12):
                lon1, lat1 = poly[i]
                lon2, lat2 = poly[i+1]
                _, _, d = geod.inv(lon1, lat1, lon2, lat2)
                print(i, '->', i+1, ':', round(d,2), 'm')
            N = len(poly)
            mid = N//2
            print('mid slice:')
            for i in [mid-3, mid-2, mid-1, mid, mid+1, mid+2]:
                print(i, poly[i])

if __name__ == '__main__':
    main()
