def get_center_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return (int((x2+x1)/2),int((y2+y1)/2))

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def measure_distance(p1,p2):
    return ((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return (p1[0]-p2[0],p1[1]-p2[1])

def get_foot_position(bbox):
    return ((bbox[0]+bbox[2])/2,(bbox[3]))