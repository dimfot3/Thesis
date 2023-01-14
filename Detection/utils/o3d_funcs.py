import numpy as np
import open3d as o3d
import time
import pandas as pd
import os
from scipy.spatial.transform import Rotation as rot_mat


def load_pcl(pcl_path):
    """
    load_pcl is loading a .pcd file as open3d Pointcloud

    :param pcl_path: the path where the .pcd is saved
    :return: open3d Pointcloud
    """
    if(pcl_path[-3:] == 'bin'):
        pcd = np.fromfile(pcl_path, dtype='float32').reshape(-1, 4)
        pcd = numpy_to_o3d(pcd)
    else:
        pcd = o3d.io.read_point_cloud(pcl_path)
    return pcd

def o3d_to_numpy(pcd):
    """
    o3d_to_numpy transforms a o3d PointCloud (3 channels) to numpy array 3D array

    :param pcd: open3d Pointcloud
    :return: numpy Nx3 ndarray
    """
    return np.array(pcd.points)

def numpy_to_o3d(pcl):
    """
    numpy_to_o3d transforms a numpy array to open3d Pointcloud

    :param pcl: numpy array Nx3d
    :return: open3d Pointcloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
    return pcd

def plot_frame(pcd):
    """
    plot_frame plots a open3D Pointcloud

    :param pcd: the open3d Pointcloud
    :return: None
    """
    o3d.visualization.draw_geometries([pcd])

def anotate_pcl_lcas(pcd, dflabels):
    """
    anotate_pcl is used for annotation of https://lcas.lincoln.ac.uk/wp/research/data-sets-software/l-cas-3d-point-cloud-people-dataset/
    It annotates a open3d Pointcloud by saving a [1, 0, 0] vector in corresponding color of Pointcloud class.

    :param pcd: the open3d Pointcloud
    :param dflabels: the open3d Pointcloud
    :return: annotated open3d Pointcloud
    """
    numpypoints = o3d_to_numpy(pcd)
    colors = np.asarray(pcd.colors)
    for i in range(dflabels.shape[0]):
        log_vec = (numpypoints[:,0]>dflabels.iloc[i]['minx']) & (numpypoints[:,0]<dflabels.iloc[i]['maxx']) & \
                  (numpypoints[:,1]>dflabels.iloc[i]['miny']) & (numpypoints[:,1]<dflabels.iloc[i]['maxy']) & \
                  (numpypoints[:,2]>dflabels.iloc[i]['minz']) & (numpypoints[:,2]<dflabels.iloc[i]['maxz'])
        colors[log_vec, 1:] = 0
        colors[log_vec, 0] = 1
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def get_3d_boxes(dflabels, points_min=300):
    """
    anotate_pcl is used for annotation of https://jrdb.erc.monash.edu/
    It returns the 3d boxes of annotated pedestrians

    :param dflabels: the labels of annotated pointclouds
    :return: the 3d boxes as list of open3d.geometry.OrientedBoundingBox
    """
    boxes3d = []
    for i in range(dflabels.shape[0]):
        if(dflabels['num_points'][i] < points_min):
            continue
        center = np.array([dflabels['cx'][i], dflabels['cy'][i], dflabels['cz'][i]])
        r = rot_mat.from_euler('z', -dflabels['rot_z'][i], degrees=False).as_matrix()
        size = np.array([dflabels['l'][i], dflabels['w'][i], dflabels['h'][i]])
        boxes3d.append(o3d.geometry.OrientedBoundingBox(center, r, size))
    return boxes3d

def get_point_annotations_kitti(pcd, dflabels, points_min=300):
    """
    anotate_pcl is used for annotation of https://jrdb.erc.monash.edu/
    It returns the 3d boxes of annotated pedestrians

    :param dflabels: the labels of annotated pointclouds
    :return: the 3d boxes as list of open3d.geometry.OrientedBoundingBox
    """
    pcl = o3d_to_numpy(pcd)
    annotations = np.zeros(pcl.shape[0], dtype='bool')
    for i in range(dflabels.shape[0]):
        if(dflabels['num_points'][i] < points_min):
            continue
        center = np.array([dflabels['cx'][i], dflabels['cy'][i], dflabels['cz'][i]])
        r = rot_mat.from_euler('z', dflabels['rot_z'][i], degrees=False).as_matrix()
        size = np.array([dflabels['l'][i], dflabels['w'][i], dflabels['h'][i]])
        box3d = np.asarray(o3d.geometry.OrientedBoundingBox(center, r, size).get_box_points())
        minx, maxx = box3d[:, 0].min(), box3d[:, 0].max()
        miny, maxy = box3d[:, 1].min(), box3d[:, 1].max()
        minz, maxz = box3d[:, 2].min(), box3d[:, 2].max()
        annotations = annotations | ((pcl[:, 0] > minx) & (pcl[:, 0] < maxx) & (pcl[:, 1] > miny) & \
             (pcl[:, 1] < maxy) & (pcl[:, 2] > minz) & (pcl[:, 2] < maxz))
    return annotations

def plot_animation_lcas(path, lebels_path, frame_pause=0.5):
    """
    plot_animation is used for https://lcas.lincoln.ac.uk/wp/research/data-sets-software/l-cas-3d-point-cloud-people-dataset/.
    It plays an animation of pointclouds and their human annotations.

    :param path: the path to folder of .pcd
    :param lebels_path: the path to folder with lebels
    :param frame_pause: the time in seconds that animation should stop after each frame (default:0.5)
    :return: None
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)
    view_rot = rot_mat.from_euler('x', 55, degrees=True).as_matrix()
    pcd_arr = np.array([str for str in os.listdir(path)])
    pcd_labels = np.array([str for str in os.listdir(lebels_path)])
    for pcd_path in pcd_arr:
        pcd = o3d.io.read_point_cloud(path+pcd_path)
        columns = ['type', 'cx', 'cy', 'cz', 'minx', 'miny', 'minz', 'maxx', 'maxy', 'maxz', 'vis']
        labels = pd.DataFrame(columns =columns) if pcd_path.replace('pcd', 'txt') not in pcd_labels else \
        pd.read_csv(lebels_path+pcd_path.replace('pcd', 'txt'), sep=' ', names=columns)
        center_dits = np.linalg.norm(o3d_to_numpy(pcd), axis=1).reshape(-1, 1)
        geometry.points = pcd.points
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(center_dits/center_dits.max(), 3, axis=1))
        pcd_anotated = anotate_pcl_lcas(pcd, labels)
        vis.add_geometry(pcd_anotated)
        geometry.colors = pcd_anotated.colors
        view_control = vis.get_view_control()
        view_control.set_lookat(np.dot(view_rot, [0,0,1]))
        view_control.set_front(np.dot(view_rot, [0,0,1]))
        view_control.set_up(np.dot(view_rot, [0,1,0]))
        view_control.set_zoom(0.1)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(frame_pause)
        vis.remove_geometry(pcd, False)
    vis.destroy_window()


def plot_animation_kitti(pcd_path, labels_path, frame_pause=0.5, box=True, frame_to_plot=1):
    """
    plot_animation is used for https://lcas.lincoln.ac.uk/wp/research/data-sets-software/l-cas-3d-point-cloud-people-dataset/.
    It plays an animation of pointclouds and their human annotations as 3d boxes.

    :param pcd_path: the path to folder of .pcd
    :param lebels_path: the path to folder with annotations (.txt)
    :param frame_pause: the time in seconds that animation should stop after each frame (default:0.5)
    :param box: if True the annotations are represented as box else it draws the points
    :return: None
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)
    view_rot = rot_mat.from_euler('x', 55, degrees=True).as_matrix()
    pcd_files = np.array([str.replace('txt', 'bin') for str in os.listdir(labels_path)])[:frame_to_plot]
    for pcd_file in pcd_files:
        pcd = load_pcl(pcd_path+pcd_file)
        columns = ['obs_angle', 'l', 'w', 'h', 'cx', 'cy', 'cz', 'rot_z', 'num_points']
        labels = pd.read_csv(labels_path + pcd_file.replace('bin', 'txt'), sep=' ', header=None, names=columns)
        boxes3d = get_3d_boxes(labels)
        vis.add_geometry(pcd)
        if box:
            for box3d in boxes3d:
                vis.add_geometry(box3d)
        else:
            annotations = get_point_annotations_kitti(pcd, labels, points_min=200)
            colors = np.zeros((len(pcd.points), 3))
            colors[annotations] = np.array([1, 0, 0])
            colors[~annotations] = np.array([0, 0, 1])
            pcd.colors = o3d.utility.Vector3dVector(colors)
        view_control = vis.get_view_control()
        view_control.set_lookat(np.dot(view_rot, [0,0,1]))
        view_control.set_front(np.dot(view_rot, [0,0,1]))
        view_control.set_up(np.dot(view_rot, [0,1,0]))
        view_control.set_zoom(0.1)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(frame_pause)
        vis.remove_geometry(pcd, False)
        if box:
            for box3d in boxes3d:
                vis.remove_geometry(box3d, False)
    vis.destroy_window()


def plot_frame_annotation_kitti(pcl_file, labels_file, box=True):
    """
    plot_frame_annotation is used for https://jrdb.erc.monash.edu/
    It plots a single frame pcl with its annotations

    :param pcl_file: the pointcloud (.bin) file
    :param labels_file: the label file
    :param box: if True prints the annotations in 3d box format else prints the points
    :return: None
    """
    annotations = []
    pcd = load_pcl(pcl_file)
    columns = ['obs_angle', 'l', 'w', 'h', 'cx', 'cy', 'cz', 'rot_z', 'num_points']
    labels = pd.read_csv(labels_file, sep=' ', header=None, names=columns)
    if box:
        boxes3d = get_3d_boxes(labels)
        o3d.visualization.draw_geometries([pcd, *boxes3d])
    else:
        annotations = get_point_annotations_kitti(pcd, labels, points_min=400)
        colors = np.zeros((len(pcd.points), 3))
        colors[annotations] = np.array([1, 0, 0])
        colors[~annotations] = np.array([0, 0, 1])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])