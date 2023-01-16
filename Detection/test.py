import numpy as np
import open3d as o3d
from utils.o3d_funcs import *
import os 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R


#plot_animation_kitti('./datasets/JRDB/velodyne/', './datasets/JRDB/labels/', frame_to_plot=10)
plot_frame_annotation_kitti('./datasets/JRDB/velodyne/000006.bin', './datasets/JRDB/labels/000006.txt', False)
# image = first_person_plot_kitti('./datasets/JRDB/velodyne/000000.bin', './datasets/JRDB/labels/# 000000.txt', fov_up=15.0, fov_down=-15.0, proj_H=20, proj_W=500, max_range=20)
# plt.imshow(image)
# plt.show()


