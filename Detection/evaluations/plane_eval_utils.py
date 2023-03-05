import struct
from typing import List
import numpy as np
import sys
sys.path.insert(0, '../utils')
from o3d_funcs import numpy_to_o3d, plot_frame, pcl_voxel, o3d_to_numpy, load_pcl
import matplotlib.pyplot as plt


class Plane:
    def __init__(self):
        self.normal = None
        self.inliers = []

def readPlanes(file: str):
    planes = []
    with open(file, 'rb') as f:
        # Read the number of circles and planes
        numPlanes, = struct.unpack('Q', f.read(8))
        # Read each plane
        for i in range(numPlanes):
            plane = Plane()
            plane.normal = struct.unpack('<3f', f.read(12))
            numInliers = struct.unpack('<Q', f.read(8))[0]
            format_str = '{}Q'.format(numInliers)
            plane.inliers = list(struct.unpack(format_str, f.read(numInliers * 8)))
            planes.append(plane)
    return planes

def save_planes(planes, fileout):
    num_of_planes = len(planes)
    with open(fileout, 'wb') as f:
       data = struct.pack('Q', num_of_planes)
       f.write(data)
       for plane in planes:
            num_inliers = len(plane.inliers)
            format_str = '<3f {}Q'.format(num_inliers + 1)
            data = struct.pack(format_str, *plane.normal, num_inliers, *plane.inliers)
            f.write(data)

def get_valid_points(points, line_points):
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    if x1 == x2: # vertical line, all points are valid
        return points
    m = (y2 - y1) / (x2 - x1) # slope of the line
    b = y1 - m * x1 # y-intercept of the line
    x_min, x_max = min(x1, x2), max(x1, x2) # minimum and maximum x-coordinates of the line
    valid_points = np.zeros(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        x, y = point
        if x_min-0.5 <= x <= x_max+0.5: # point lies on the line segment
            v_x = (x + m * y - m * b) / (m**2 + 1) # x-coordinate of the point where the line passing through the point is vertical to the line defined by line_points
            if x_min-0.5 <= v_x <= x_max+0.5: # point lies between the vertical lines passing through line_points
                valid_points[i] = True
    return valid_points

def match_points_with_line(points, lines_points):
    dists_accum = np.zeros((points.shape[0], lines_points.shape[0] // 2), dtype='float')
    for i in range(lines_points.shape[0] // 2):
        dists_accum[:, i] = get_dists_from_line(points, lines_points[2*i:2*i+2, :2])
        valid_points = get_valid_points(points, lines_points[2*i:2*i+2, :2])
        dists_accum[~valid_points, i] = np.inf
    point_class = np.argmin(dists_accum, axis=1)
    return point_class

def get_dists_from_line(points, line_points):
    # Unpack the two points defining the line
    x1, y1 = line_points[0]
    x2, y2 = line_points[1]
    # Calculate the coefficients of the line equation
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    # Calculate the distance between each point and the line
    dists = np.zeros((points.shape[0], ))
    for i, point in enumerate(points):
        x, y = point
        dists[i]= np.abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
    return dists

def normals_from_lines(lines_point):
    normal_vecs = np.zeros((lines_point.shape[0] // 2, 3))
    for i in range(normal_vecs.shape[0]):
        direction_vector = lines_point[2 * i] - lines_point[2 * i + 1]
        normal_vecs[i] = np.array([-direction_vector[1], direction_vector[0], 0])
        normal_vecs[i] = normal_vecs[i] / np.linalg.norm(normal_vecs[i])
    return normal_vecs

def create_planes(points, lines_points):
    point_class = match_points_with_line(points, lines_points)
    normals = normals_from_lines(lines_points)

    return point_class, normals

def create_planes(points, lines_points, initil_idxs):
    point_class = match_points_with_line(points, lines_points)
    normals = normals_from_lines(lines_points)
    planes = []
    for i in range(normals.shape[0]):
        plane = Plane()
        plane.normal = normals[i]
        plane.inliers = initil_idxs[np.argwhere(point_class == i)[:, 0].reshape(-1, )]
        planes.append(plane)
    return planes

# data = np.load('./plane_detection_dataset/plane_annot_jrdb3.npz', allow_pickle=True)
# pcl = o3d_to_numpy(load_pcl('./pointclouds/jrdb3.bin'))
# planes = create_planes(pcl[data['wall_points_idxs'], :2], data['line_points'], data['wall_points_idxs'])
# ax = plt.subplot(1, 1, 1, projection='3d')
# save_planes(planes, 'jrdb_ground_truth.bin')

# for cls in np.unique(point_class):
#     idxs = data['wall_points_idxs'][point_class==cls]
#     ax.scatter(pcl[idxs, 0], pcl[idxs, 1], pcl[idxs, 2])
#     ax.quiver(pcl[idxs, 0].mean(), pcl[idxs, 1].mean(), pcl[idxs, 2].mean(), normals[cls, 0], normals[cls, 1], normals[cls, 2], length=3, normalize=True)
# plt.show()

# pcl = o3d_to_numpy(load_pcl(f'./pointclouds/jrdb3.bin'))
# planes = create_planes(pcl[data['wall_points_idxs'], :2], data['line_points'])
# save_planes(planes, 'jrdb3_ground_truth.bin')

"""
files = ['jrdb3', 'computer']
for file in files:
    pcl = o3d_to_numpy(load_pcl(f'./pointclouds/{file}.bin'))
    planes = readPlanes(f'./pointclouds/{file}_ground_truth.bin')
    
    ax = plt.subplot(1, 1, 1, projection='3d')
    for i, plane in enumerate(planes):
        ax.scatter(pcl[plane.inliers, 0], pcl[plane.inliers, 1], pcl[plane.inliers, 2])
        ax.quiver(pcl[plane.inliers, 0].mean(), pcl[plane.inliers, 1].mean(), pcl[plane.inliers, 2].mean(), plane.normal[0], plane.normal[1], plane.normal[2], length=1, normalize=True)
    plt.show()
"""
