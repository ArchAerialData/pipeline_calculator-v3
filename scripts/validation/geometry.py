"""Independent, small-fixture local-plane polygon checks for validation reports."""
import math


def local_points(points, origin):
    from pyproj import Geod
    geod=Geod(ellps='GRS80')
    result=[]
    for point in points:
        az,_,distance=geod.inv(*origin,*point)
        result.append((distance*math.sin(math.radians(az)),distance*math.cos(math.radians(az))))
    return result


def segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    denominator=dx*dx+dy*dy
    t=0 if denominator==0 else max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/denominator))
    return math.dist(p,(a[0]+t*dx,a[1]+t*dy))


def inside(p,ring):
    hit=False
    for a,b in zip(ring,ring[1:]):
        if segment_distance(p,a,b)<1e-7:
            return True
        if (a[1]>p[1])!=(b[1]>p[1]):
            cross_x=a[0]+(p[1]-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if p[0]<cross_x:
                hit=not hit
    return hit


def ring_checks(ring):
    # Remove consecutive coincident vertices before testing nonadjacent edges.
    # Near-zero joins in raw floating-point geometry are not polygon crossings.
    cleaned=[]
    for point in ring:
        if not cleaned or math.dist(point,cleaned[-1])>1e-7:
            cleaned.append(point)
    if cleaned and math.dist(cleaned[0],cleaned[-1])<=1e-7:
        cleaned[-1]=cleaned[0]
    ring=cleaned
    if len(ring)>4096:
        raise ValueError('Gallery topology check limited to 4096 vertices')
    area=abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(ring,ring[1:]))/2)
    intersections=0
    pairs=list(zip(ring,ring[1:]))
    for i,(a,b) in enumerate(pairs):
        for j,(c,d) in enumerate(pairs[i+2:],i+2):
            if i==0 and j==len(pairs)-1:
                continue
            # Solve the segment parameters independently of the production checker.
            u=(b[0]-a[0],b[1]-a[1]);v=(d[0]-c[0],d[1]-c[1]);w=(c[0]-a[0],c[1]-a[1])
            determinant=u[0]*v[1]-u[1]*v[0]
            if abs(determinant)>1e-10:
                t=(w[0]*v[1]-w[1]*v[0])/determinant
                s=(w[0]*u[1]-w[1]*u[0])/determinant
                intersections += -1e-9<=t<=1+1e-9 and -1e-9<=s<=1+1e-9
            elif min(segment_distance(c,a,b),segment_distance(d,a,b),segment_distance(a,c,d),segment_distance(b,c,d))<1e-8:
                intersections+=1
    return {'closed':bool(ring) and ring[0]==ring[-1], 'area_m2':area,
            'self_intersections':intersections,'finite':all(math.isfinite(v) for p in ring for v in p)}
