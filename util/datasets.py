import argparse
import os
import pickle
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import PIL
from PIL import Image

from torchvision import transforms, datasets


class ImgWithPickledBoxesDataset(datasets.VisionDataset):
    def __init__(self, root, transform):
        super(ImgWithPickledBoxesDataset, self).__init__(root, transform=transform)

        self.root = root
        self.transform = transform

        self.file_list = [filename for filename in os.listdir(self.root) if filename.endswith('.png')]

    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.root, filename)
        pickle_path = os.path.join(self.root, os.path.splitext(filename)[0] + '.pkl')

        image = Image.open(image_path).convert('RGB')
        with open(pickle_path, 'rb') as pickle_file:
            pickle_data = pickle.load(pickle_file)

        return self.transform(image, pickle_data)

    def __len__(self):
        return len(self.file_list)
