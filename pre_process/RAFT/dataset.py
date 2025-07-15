import os

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

class RAFTDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.data_points = self._find_data_points()

    def __len__(self):
        return len(self.data_points) - 1

    def __getitem__(self, idx):
        image_path = self.data_points[idx]
        next_image_path = self.data_points[idx + 1]
        
        image = decode_image(image_path)
        next_image = decode_image(next_image_path)

        if self.transforms:
            image = self.transforms(image)
            next_image = self.transforms(next_image)

        return image, next_image

    def _find_data_points(self):
        image_dir = os.path.join(self.root_dir, 'images')
        
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        data_points = []
        image_filenames = sorted(os.listdir(image_dir))
        
        for img_name in image_filenames:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, img_name)
                data_points.append({'image_path': image_path})

        return data_points