import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors

class Touchstone3DDataset(Dataset):
    """
    Dataset class for the Touchstone3D dataset.
    """
    def __init__(self, map_file, npts, mode, data_path, pc_attribs='xyz', pc_augm=False, pc_augm_config=None):
        """
        Args:
            map_file (str): Path to the class mapping file.
            npts (int): Number of points to sample from each point cloud.
            mode (str): Dataset mode ('train' or 'test').
            data_path (str): Path to the directory containing the data blocks.
            pc_attribs (str): Point cloud attributes to use (e.g., 'xyz', 'xyzrgb').
            pc_augm (bool): Whether to apply data augmentation.
            pc_augm_config (dict): Configuration for data augmentation.
        """
        self.npts = npts
        self.mode = mode
        self.data_path = data_path
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config if pc_augm_config is not None else {}

        # Load class mapping
        with open(map_file, 'rb') as f:
            self.class_map = pickle.load(f)
        self.CLASS_LABELS = list(self.class_map.keys())
        self.CLASS2COLOR = self.class_map  # Store the color mapping

        # Load data files
        self.data_files = sorted(os.listdir(data_path))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load point cloud and labels
        file_path = os.path.join(self.data_path, self.data_files[idx])
        data = np.load(file_path)
        points = data['points']  # Shape: (N, 3)
        labels = data['labels']  # Shape: (N,)

        # Sample fixed number of points
        if points.shape[0] > self.npts:
            idxs = np.random.choice(points.shape[0], self.npts, replace=False)
        else:
            idxs = np.random.choice(points.shape[0], self.npts, replace=True)
        points = points[idxs]
        labels = labels[idxs]

        # Data augmentation
        if self.mode == 'train' and self.pc_augm:
            points = self.augment_point_cloud(points)

        # Normalize points to zero mean and unit variance
        points = self.normalize_points(points)

        # Convert to torch tensors
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        return points, labels

    def normalize_points(self, points):
        """
        Normalize point cloud to zero mean and unit variance.
        """
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
        points /= furthest_distance
        return points

    def augment_point_cloud(self, points):
        """
        Apply data augmentation to the point cloud.
        """
        # Random scaling
        if 'scale' in self.pc_augm_config:
            scale = np.random.uniform(self.pc_augm_config['scale'][0], self.pc_augm_config['scale'][1])
            points *= scale

        # Random rotation
        if 'rot' in self.pc_augm_config:
            angle = np.random.uniform(-self.pc_augm_config['rot'], self.pc_augm_config['rot'])
            cosval = np.cos(angle)
            sinval = np.sin(angle)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                       [sinval, cosval, 0],
                                       [0, 0, 1]])
            points = np.dot(points, rotation_matrix)

        # Random jittering
        if 'jitter' in self.pc_augm_config:
            jitter = np.random.normal(0, self.pc_augm_config['jitter'], size=points.shape)
            points += jitter

        return points