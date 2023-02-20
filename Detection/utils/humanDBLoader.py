import torch
import os
from torch.utils.data import Dataset, DataLoader
from o3d_funcs import load_pcl, o3d_to_numpy, \
get_point_annotations_kitti, pcl_voxel, split_3d_point_cloud_overlapping, plot_frame_annotation_kitti_v2
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data_path, pcl_len=2048):
        pcl_files = [file.replace('.bin', '') for file in os.listdir(data_path + 'velodyne/')]
        label_files = [file.replace('.txt', '') for file in os.listdir(data_path + 'labels/')]
        labeled_pcls = np.intersect1d(pcl_files, label_files)
        self.data_path = data_path
        self.pcl_files = [file + '.bin' for file in labeled_pcls]
        self.label_files = [file + '.txt' for file in labeled_pcls]
        self.label_cols = ['obs_angle', 'l', 'w', 'h', 'cx', 'cy', 'cz', 'rot_z', 'num_points']
        self.pcl_len = pcl_len

    def __len__(self):
        return len(self.pcl_files)

    def __getitem__(self, idx):
        # load pcl and its annotations
        pcl = load_pcl(self.data_path + 'velodyne/' + self.pcl_files[idx])
        labels = pd.read_csv(self.data_path + 'labels/' + self.label_files[idx], sep=' ', header=None, names=self.label_cols)

        # transform the pcl and annotations
        # voxel downsample
        pcl_voxeled = pcl_voxel(pcl, voxel_size=0.1)
        annotations = get_point_annotations_kitti(pcl_voxeled, labels, points_min=100)
        pcl_numpy = o3d_to_numpy(pcl_voxeled)
        # 3d tiling
        splitted_pcl, splitted_ann = split_3d_point_cloud_overlapping(pcl_numpy, annotations, 6, 0.3)
        return splitted_pcl, splitted_ann


if __name__ == '__main__':
    path = '/home/fdimitri/workspace/Thesis/Detection/datasets/JRDB/'
    dataset = MyDataset(path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for pcls, annotations in dataset:
        for i, pcl in enumerate(pcls):
            plot_frame_annotation_kitti_v2(pcl[0], annotations[i])
        exit()
