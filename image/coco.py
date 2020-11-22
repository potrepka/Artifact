import common.data as data
from image.image import ImageReader
import numpy as np
import os
import random
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def paths(directory):
    walk = os.walk(directory)
    paths = []
    for dir in walk:
        for file in dir[2]:
            path = os.path.join(dir[0], file)
            paths.append(path)
    return paths

class COCO(Dataset):
    def __init__(self, directory_a, directory_b, num_training=None,
                 num_validation=None, size=256, channels=3, shuffle=False,
                 cache=False, validation=False):
        self.paths_a = paths(directory_a)
        self.paths_b = paths(directory_b)
        self.length_ab = min(len(self.paths_a), len(self.paths_b))
        if num_training is None:
            length_training = self.length_ab
        else:
            num_training = max(num_training, 0)
            length_training = min(num_training, self.length_ab)
        if num_validation is None:
            length_validation = self.length_ab
        else:
            num_validation = max(num_validation, 0)
            length_validation = min(num_validation, self.length_ab)
        if validation:
            self.length = length_validation
            self.begin = max(self.length_ab, 1) - max(self.length, 1)
        else:
            self.length = length_training
            self.range = max(self.length_ab - length_validation, 1)
        self.read_image = ImageReader(size, channels)
        self.image_a = {}
        self.image_b = {}
        self.directory_a = directory_a
        self.directory_b = directory_b
        self.num_training = num_training
        self.num_validation = num_validation
        self.size = size
        self.channels = channels
        self.shuffle = shuffle
        self.cache = cache
        self.validation = validation

    def __getitem__(self, i):
        if self.length == 0:
            image_a = torch.zeros(self.channels, self.size, self.size)
            image_b = torch.zeros(self.channels, self.size, self.size)
            return image_a, image_b
        if self.shuffle:
            a = self.__getindex__(i)
            b = self.__getindex__(i)
            path_a = self.paths_a[a]
            path_b = self.paths_b[b]
        else:
            a = self.__getindex__(i)
            b = a
            path_a = self.paths_a[a]
            path_b = self.paths_b[b]
            name_a = os.path.splitext(os.path.basename(path_a))[0]
            name_b = os.path.splitext(os.path.basename(path_b))[0]
            if name_a != name_b:
                raise ValueError('File names must match when shuffle is off.')
        if a in self.image_a:
            image_a = self.image_a[a]
        else:
            image_a = self.read_image(path_a)
            if self.cache:
                self.image_a[a] = image_a
        if b in self.image_b:
            image_b = self.image_b[b]
        else:
            image_b = self.read_image(path_b)
            if self.cache:
                self.image_b[b] = image_b
        return image_a, image_b

    def __len__(self):
        return max(self.length, 1)

    def __getindex__(self, i):
        if self.validation:
            return self.begin + i
        else:
            return random.randrange(self.range)
