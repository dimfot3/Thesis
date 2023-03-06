import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class MNIST3D(Dataset):
    """3D MNIST dataset."""
    
    NUM_CLASSIFICATION_CLASSES = 10
    POINT_DIMENSION = 3

    def __init__(self, num_points, root_path='./data'):
        self.dataset = MNIST(root=root_path, train=True, download=True)
        self.number_of_points = num_points

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img,label = self.dataset[idx]
        pc = self.transform_img2pc(img)
        
        if self.number_of_points-pc.shape[0]>0:
            # Duplicate points
            sampling_indices = np.random.choice(pc.shape[0], self.number_of_points-pc.shape[0])
            new_points = pc[sampling_indices, :]
            pc = np.concatenate((pc, new_points),axis=0)
        else:
            # sample points
            sampling_indices = np.random.choice(pc.shape[0], self.number_of_points)
            pc = pc[sampling_indices, :]
            
        pc = pc.astype(np.float32)
        # add z
        noise = np.random.normal(0,0.05,len(pc))
        noise = np.expand_dims(noise, 1)
        pc = np.hstack([pc, noise]).astype(np.float32)
        pc = torch.tensor(pc)
        
        return pc, label
    
    def transform_img2pc(self, img):
        img_array = np.asarray(img)
        indices = np.argwhere(img_array > 127)
        return indices.astype(np.float32)


if __name__ == '__main__':
    dataset_3d = MNIST3D(1024)
    l_data = len(dataset_3d)
    batch_size = 3
    test_dataloader = DataLoader(dataset_3d, batch_size=batch_size, shuffle=False)
    for data, labels in test_dataloader:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        for batch in range(batch_size):
            ax = fig.add_subplot(1, batch_size, batch + 1, projection='3d')
            ax.scatter3D(data[batch, :, 0], data[batch, :, 1], data[batch, :, 2], color = "green")
            ax.set_title(f'label: {labels[batch]}')
        plt.show()
        
    