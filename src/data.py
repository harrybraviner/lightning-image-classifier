"""
Contains pytorch Datasets for loading images from the animals10 dataset, and for cropping from an existing dataset.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class AnimalsDataset(Dataset):

    def __init__(
        self,
        data_root: str,
        is_test: bool=False,
    ):
        self.root_dir = os.path.expanduser(data_root)
        self.is_test = is_test

        self.dataframe = pd.read_csv(os.path.join(data_root, 'table.csv'))

        # Convert labels to indices
        categories = sorted(list(set(self.dataframe['category'])))
        
        # Discard now all of the examples that we don't want to use.
        self.dataframe = self.dataframe[self.dataframe['is_test'] == is_test]
        self.dataframe.reset_index(drop=True, inplace=True)
        self.dataframe['y_idx'] = self.dataframe['category'].map(lambda c: categories.index(c))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_filename = os.path.join(self.root_dir, self.dataframe['filename'][idx])
        image = Image.open(img_filename)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        y_idx = self.dataframe['y_idx'][idx]
        
        return image, y_idx

class RandomCropDataset(Dataset):

    def __init__(self, dataset, post_crop_transform):
        self.inner_dataset = dataset
        self.post_crop_transform = post_crop_transform

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        image, y_idx = self.inner_dataset[idx]

        # Crop the image to be square
        width = image.width
        height = image.height
        if height < width:
            # Image wider than it is tall, will crop width.
            width_desired = height
            x = torch.rand(())
            l = int(x*(width - width_desired))
            r = l + width_desired
            t = 0
            b = height
            image = image.crop((l, t, r, b))
        elif width < height:
            # Image is taller than it is wide, will crop height
            height_desired = width
            x = torch.rand(())
            l = 0
            r = width
            t = int(0.5*(height - height_desired))
            b = t + height_desired
            image = image.crop((l, t, r, b))

        image = self.post_crop_transform(image)
        
        return image, y_idx

class CenterCropDataset(Dataset):

    def __init__(self, dataset, post_crop_transform):
        self.inner_dataset = dataset
        self.post_crop_transform = post_crop_transform

    def __len__(self):
        return len(self.inner_dataset)

    def __getitem__(self, idx):
        image, y_idx = self.inner_dataset[idx]

        # Crop the image to be square
        width = image.width
        height = image.height
        if height < width:
            # Image wider than it is tall, will crop width.
            width_desired = height
            x = 0.5
            l = int(x*(width - width_desired))
            r = l + width_desired
            t = 0
            b = height
            image = image.crop((l, t, r, b))
        elif width < height:
            # Image is taller than it is wide, will crop height
            height_desired = width
            x = 0.5
            l = 0
            r = width
            t = int(0.5*(height - height_desired))
            b = t + height_desired
            image = image.crop((l, t, r, b))

        image = self.post_crop_transform(image)

        return image, y_idx
