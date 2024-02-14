from torch.utils.data import Dataset as TorchBaseDataset
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import glob
from random import choice
import os
import matplotlib.pyplot as plt

class ImageDataset(TorchBaseDataset):
    def __init__(self, fpath: str=None):
        if fpath is not None:
            self.image_paths = glob.glob(os.path.join(fpath, '*.{jpg, png}'))
        
        self.augmentations = []
        self.inverse_augmentations = []
        
    def add_augmentations(self, augmentations: list=[]):
        self.augmentations = augmentations  

    def add_inverse_augmentations(self, inverse_augmentations: list=[]):
        self.inverse_augmentations = inverse_augmentations

    def invert_augmentations(self, image):
        for aug in self.inverse_augmentations:
            image = aug(image)
        return image
    
    def __len__(self):
        return len(glob.glob(os.path.join(self.data_file_path, '*.jpg')))
    
    def __getitem__(self, _):
        self.image = np.array(Image.open(choice(self.image_paths)))

        if self.augmentations is not None:
            for aug in self.augmentations:
                self.image = aug(self.image)
        return self.image
    
    def display_example(self):
        self.inverse_augmentations(self.__getitem__(0))