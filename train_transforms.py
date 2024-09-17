import os
import sys
import torch
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'prov-gigapath')))
from finetune.datasets.slide_datatset import SlideDataset

class SlideDatasetWithTransforms(SlideDataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 splits: list,
                 task_config: dict,
                 slide_key='slide_id',
                 split_key='pat_id',
                 cropping_ratio=0.875,  # New argument for cropping ratio
                 flip_prob=0.5,         # Probability for horizontal flip
                 noise_std=0.01,        # Standard deviation for Gaussian noise
                 shift_scale=5,         # Tile shifting scale
                 grid_size=1000,        # Size of the grid for shifting
                 tile_size=256,         # Tile size for shifting
                 **kwargs):
        # Call the parent class constructor
        super(SlideDatasetWithTransforms, self).__init__(data_df, root_path, splits, task_config, slide_key, split_key, **kwargs)
        
        # Store additional transformation parameters
        self.cropping_ratio = cropping_ratio
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.shift_scale = shift_scale
        self.grid_size = grid_size
        self.tile_size = tile_size


    def __getitem__(self, idx):
        # Get the sample using the parent method
        sample = super(SlideDatasetWithTransforms, self).__getitem__(idx)
        sample = self.apply_transforms(sample)  # Apply the transformations
        return sample

    def apply_transforms(self, sample):
        '''Apply a sequence of transformations to the sample'''
        data = {'imgs': sample['imgs'], 'coords': sample['coords']}                       
        data = random_resized_crop(data, self.cropping_ratio)                                                       # Random resized crop
        data['coords'] = random_shift_coords(data['coords'], self.grid_size, self.tile_size, self.shift_scale)      # Random shift
        data['coords'] = random_horizontal_flip(data['coords'], self.flip_prob)                                     # Random horizontal flip
        data['imgs'] = add_gaussian_noise(data['imgs'], self.noise_std)                                             # Add Gaussian noise

        # Update sample with transformed data
        sample['imgs'] = data['imgs']
        sample['coords'] = data['coords']

        return sample

"""
TRANSFORM FUNCTIONS (NEED TO BE UNIT TESTED)
"""

def random_resized_crop(data, cropping_ratio):
    imgs = data['imgs']  # Shape: [embed_size, D]
    coords = data['coords']  # Shape: [embed_size, 2]

    # Reorder by x-coordinate (ascending order)
    embed_size, D = imgs.shape 
    
    # Sort by x-coordinate of `coords`
    coords_sorted_idx = torch.argsort(coords[:, 0], stable=True)
    
    # Reorder imgs and coords according to sorted indices
    coords_sorted = coords[coords_sorted_idx]
    imgs_sorted = imgs[coords_sorted_idx]

    # Randomly select the cropping region
    crop_size = int(embed_size * cropping_ratio)
    start_idx = torch.randint(0, embed_size - crop_size + 1, (1,)).item()  # Single random start index
    end_idx = start_idx + crop_size

    # Crop the sorted imgs and coords
    cropped_imgs = imgs_sorted[start_idx:end_idx]
    cropped_coords = coords_sorted[start_idx:end_idx]

    # Return the cropped data
    cropped_data = {
        'imgs': cropped_imgs,       # Cropped embeddings [crop_size, D]
        'coords': cropped_coords    # Cropped coordinates [crop_size, 2]
    }
    
    return cropped_data

def random_shift_coords(coords, grid_size=1000, tile_size=256, shift_scale=5):
    # Generate random shifts uniformly in the range [-tile_size/2, tile_size/2] for each coordinate
    random_bias = (torch.rand_like(coords) - 0.5) * tile_size 
    shifted_coords = coords + random_bias

    # Ensure the shifted coordinates stay within the bounds of the grid
    min_coord = torch.zeros_like(coords)                                        # Minimum valid coordinate is 0
    max_coord = torch.ones_like(coords) * (grid_size * tile_size)               # Maximum valid coordinate is grid_size * tile_size

    # Clamp the shifted coordinates to ensure they stay within the valid range
    shifted_coords = torch.clamp(shifted_coords, min=min_coord, max=max_coord)
    return shifted_coords

def random_horizontal_flip(coords, flip_prob=0.5):
    max_x = torch.max(coords[:, 0]).item()  # Get the max x-coordinate
    img_width = max_x + 1
    
    if torch.rand(1).item() < flip_prob:  # Apply flip with probability flip_prob
        flipped_coords = coords.clone()
        flipped_coords[:, 0] = img_width - coords[:, 0]  # Flip relative to image width
        return flipped_coords
    return coords

def add_gaussian_noise(imgs, noise_std=0.01):
    noise = torch.randn_like(imgs) * noise_std
    noisy_imgs = imgs + noise
    return noisy_imgs
