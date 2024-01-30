from torch.utils.data import Dataset as TorchBaseDataset
import glob
import os

class ImageDataset(TorchBaseDataset):
    def __init__(self, fpath):
        augmentations = []
        
    def transformations():