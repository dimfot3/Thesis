import open3d as o3d
from utils.o3d_funcs import load_pcl, o3d_to_numpy, pcl_voxel
import numpy as np
import matplotlib.pyplot as plt
from utils.plane_detector import *
from sklearn.metrics import f1_score
import optuna

def objective(trial, list_of_data):
    pca_radius = trial.suggest_float('pca_radius', 0.1, 1.5)
    distmin = trial.suggest_float('distmin', 0.1, 1)
    minp = trial.suggest_int('minp', 10, 200)
    voxelm = trial.suggest_float('pcl_voxel', 0.001, 1.2)
    total_score = 0
    for data in list_of_data:
        pcl = o3d_to_numpy(pcl_voxel(load_pcl(f'./datasets/JRDB/velodyne/{data}.bin'), voxelm))
        pcl = pcl[np.linalg.norm(pcl, axis=1) < 30]
        lines = read_gr_lines(f"./datasets/JRDB/planes/{data}.dxf")
        points_wall_gr = get_wall_points(pcl, lines, 0.17)
        regions = pca_plane_det(pcl, pca_radius=pca_radius, distmin=distmin, minp=minp)
        detected_wall_points = np.zeros((pcl.shape[0],)).astype('bool')
        for region in regions:
            detected_wall_points[region] = True
        total_score += f1_score(points_wall_gr.astype('int'), detected_wall_points.astype('int'), average='binary', pos_label=1)
    return total_score

study = optuna.create_study()
files = ['000003']
study.optimize(lambda trial: objective(trial, files), n_trials=200)
print(study.best_params)
