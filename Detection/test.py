import numpy as np
import open3d as o3d
from utils.o3d_funcs import *
import os 
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from sklearn.neighbors import NearestNeighbors
import open3d.cuda as o3d_cuda
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler


def compute_local_pca(pointcloud, radius):
    """Compute the PCA of a local area around each point in the point cloud."""
    # Build a KD-tree for fast nearest neighbor search
    tree = KDTree(pointcloud[:, :3])

    # Compute the PCA for each point
    num_points = pointcloud.shape[0]
    eigenvectors = np.zeros((num_points, 3))
    eigenvalues = np.zeros((num_points, ))
    for i in range(num_points):
        # Find the nearest neighbors within a certain radius
        neighbors_idx = tree.query_radius(pointcloud[i, :3].reshape(1, -1), r=radius)[0]
        if(len(neighbors_idx) < 20):
            continue
        # Extract the local point cloud and center it around the current point
        local_points = pointcloud[neighbors_idx, :3] - pointcloud[i, :3]

        # Compute the PCA of the local point cloud
        pca = PCA(n_components=3)
        pca.fit(local_points)
        normal_verctor = pca.components_[np.argmin(pca.explained_variance_)]
        # Store the eigenvectors and eigenvalues of the PCA
        dot_product = np.dot(normal_verctor - pointcloud[i, :3], normal_verctor)
        if dot_product < 0:
            normal_verctor *= -1.0
        eigenvectors[i, :] = normal_verctor
        eigenvalues[i] = np.min(pca.explained_variance_)
    return eigenvectors, eigenvalues


def classify_points(point_vectors, eigenvalues, theta_xy_t = 45, theta_ceil_t=15, eigenvalmin=0.1):
    normals = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    valid_idxs = (point_vectors.sum(axis=1) != 0) & (eigenvalues < eigenvalmin)
    dot_prods = np.zeros((point_vectors.shape[0], normals.shape[0]))
    dot_prods[valid_idxs] = np.dot(point_vectors[valid_idxs], normals.T)
    thetas = np.arccos(dot_prods)
    min_thetas = thetas.min(axis=1)
    min_thetas_idxs = np.argmin(thetas, axis=1)
    class_arr = np.zeros((point_vectors.shape[0], ), dtype='int')
    class_arr[~valid_idxs] = -1
    class_arr[(min_thetas_idxs == 0) & valid_idxs & (min_thetas < theta_ceil_t/np.pi)] = 0
    class_arr[(min_thetas_idxs == 1) & valid_idxs & (min_thetas < theta_ceil_t/np.pi)] = 1
    class_arr[((min_thetas_idxs == 2) | (min_thetas_idxs == 3)) & valid_idxs & (min_thetas < theta_xy_t/np.pi)] = 2
    class_arr[((min_thetas_idxs == 4) | (min_thetas_idxs == 5)) & valid_idxs & (min_thetas < theta_xy_t/np.pi)] = 3
    return class_arr


def generate_regions(points, classes, distx, k):
    # Build KD-tree
    tree = KDTree(points)

    # Initialize variables
    num_points = len(points)
    regions = []
    class_regions = []
    assigned_points = np.zeros(num_points, dtype=bool)

    # Loop over all points
    for i in range(num_points):
        # Create new region
        region_points = [i]
        region_class = classes[i]
        assigned_points[i] = True
        if region_class == -1:
            continue
        # Find neighboring points within distx
        neighbors = tree.query_radius(points[i][np.newaxis], r=distx)[0]

        # Loop over neighboring points
        for j in neighbors:
            # If point is already assigned to a region, skip it
            if assigned_points[j] or classes[j] == -1:
                continue

            # Check if the point is closer to any point in the region
            closest_point_index = tree.query(points[j][np.newaxis], k=1)[1][0][0]
            dist_to_closest_point = np.linalg.norm(points[j] - points[closest_point_index])
            if dist_to_closest_point <= distx:
                # If point has the same class, add it to the region
                if classes[j] == region_class:
                    region_points.append(j)
                    assigned_points[j] = True

        # If region has at least k points, add it to the list of regions
        if len(region_points) >= k:
            regions.append(region_points)
            class_regions.append(region_class)

    return regions, class_regions

def distance_from_line_vectorized(points, line_point1, line_point2):
    x, y = points[:, 0], points[:, 1]
    x1, y1 = line_point1[0], line_point1[1]
    x2, y2 = line_point2[0], line_point2[1]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    distance = np.abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
    return distance


def fit_plane_ransac(points, inlier_thresh=0.01, max_iter=100):
    """
    Fits a plane to a set of 3D points using RANSAC.

    Args:
        points: (np.ndarray) array of shape (n, 3) containing the 3D points to fit the plane to.
        inlier_thresh: (float) distance threshold for considering a point as an inlier.
        max_iter: (int) maximum number of iterations for RANSAC.

    Returns:
        (tuple) a tuple containing:
            - (np.ndarray) array of shape (3,) representing the normal vector of the fitted plane.
            - (float) scalar representing the distance of the fitted plane to the origin.
    """
    # Calculate the centroid of the input points
    centroid = np.mean(points, axis=0)

    # Calculate the vector from each point to the centroid
    vectors = points - centroid

    # Normalize the input points and vectors to improve numerical stability
    scaler = StandardScaler()
    points_norm = scaler.fit_transform(points)
    vectors_norm = scaler.transform(vectors)

    # Calculate the normal vector of the plane using RANSAC
    model_ransac = RANSACRegressor(estimator=None, min_samples=3, residual_threshold=inlier_thresh,
                                    is_data_valid=None, is_model_valid=None, max_trials=max_iter, random_state=None)
    model_ransac.fit(vectors_norm, np.zeros(vectors_norm.shape[0]))
    plane_normal = scaler.inverse_transform(model_ransac.estimator_.coef_.reshape(1, -1)).squeeze()
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    plane_distance = -np.dot(plane_normal, centroid)

    return plane_normal, plane_distance

def fit_planes_ransac(regions, points):
    """
    Fits a plane to each region using RANSAC.

    Args:
        regions: (list of lists) a list of regions, where each region is represented by a list of point indices.
        points: (np.ndarray) array of shape (n, 3) containing the 3D points.

    Returns:
        (list of tuples) a list of tuples, where each tuple contains:
            - (np.ndarray) array of shape (3,) representing the normal vector of the fitted plane.
            - (float) scalar representing the distance of the fitted plane to the origin.
    """
    planes = []

    for region in regions:
        region_points = points[region, :]
        plane_normal, plane_distance = fit_plane_ransac(region_points)
        planes.append((plane_normal, plane_distance))

    return planes

#plot_frame_annotation_kitti('./datasets/JRDB/velodyne/000006.bin', './datasets/JRDB/labels/000006.txt', False)
pcl = o3d_to_numpy(pcl_voxel(load_pcl('./datasets/JRDB/velodyne/000006.bin'), 0.3))


t0 = time()
eigv, eigenvalues = compute_local_pca(pcl, 1.2)
classes = classify_points(eigv, eigenvalues)
regions, class_regions = generate_regions(pcl, classes, 1, 20)
planes = fit_planes_ransac(regions, pcl)
t1 = time()

# plot the points and the plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, plane in enumerate(planes):
    plane_normal = plane[0]
    plane_distance = plane[1]
    # define the plane using its normal vector and distance from the origin
    d = -plane_distance / np.linalg.norm(plane_normal)
    print()
    xx, yy = np.meshgrid(np.linspace(pcl[regions[i], 0].min(axis=0), pcl[regions[i], 0].max(axis=0), 10), \
        np.linspace(pcl[regions[i], 1].min(axis=0), pcl[regions[i], 1].max(axis=0), 10))
    z = (-plane_normal[0] * xx - plane_normal[1] * yy - d) * 1.0 / plane_normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2)

plt.show()

