import math
import random

import PIL
from PIL import ImageFilter, ImageOps
import torch
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


import argparse
import os
import pickle
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import PIL
from PIL import Image

from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import draw_bounding_boxes


class DataAugmentationForSIMTraining(object):
    def __init__(self, args):
        self.args = args

        self.random_resized_crop = SingleRandomResizedCrop(args.input_size, scale=(args.crop_min, 1.0), interpolation=3)
        self.random_flip = RandomHorizontalFlip()
        self.resize = Resize(size=args.input_size, interpolation=PIL.Image.BILINEAR)
        self.to_tensor = transforms.ToTensor()

        self.color_transform1 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        ])

        self.color_transform2 = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
        ])

    def __call__(self, image, boxes):
        boxes, _ = _pad_boxes_and_classes(boxes, None, self.args.num_boxes)
        image, boxes = self.resize(image, boxes)
        boxes0 = boxes.clone()
        spatial_image1, flip1, boxes1 = self.random_flip(image, boxes)
        spatial_image2, flip2, boxes2 = self.random_flip(image, boxes)

        spatial_image1, boxes1, i1, j1, h1, w1, W = self.random_resized_crop(spatial_image1, boxes1)
        spatial_image2, boxes2, i2, j2, h2, w2, _ = self.random_resized_crop(spatial_image2, boxes2)
        color_image1 = self.color_transform1(spatial_image1)
        color_image2 = self.color_transform2(spatial_image2)

        return {
            'x0': self.to_tensor(image),
            'x1': self.to_tensor(color_image1),
            'x2': self.to_tensor(color_image2),
            'boxes0': boxes0,
            'boxes1': boxes1,
            'boxes2': boxes2
        }

    def __repr__(self):
        repr = "(DataAugmentationTraining,\n"
        repr += "  transform = %s,\n" % str(self.resize) + str(self.random_resized_crop) + str(self.random_flip) + \
                str(self.color_transform1) + str(self.to_tensor)
        repr += ")"
        return repr

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = cfg

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i1 = torch.randint(0, height - h + 1, size=(1,)).item()
                i2 = torch.randint(0, height - h + 1, size=(1,)).item()
                j1 = torch.randint(0, width - w + 1, size=(1,)).item()
                j2 = torch.randint(0, width - w + 1, size=(1,)).item()

                return i1, j1, i2, j2, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i1, j1, i2, j2, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i1, j1, h, w, self.size, self.interpolation), \
            F.resized_crop(img, i2, j2, h, w, self.size, self.interpolation), (i2-i1)/h, (j2-j1)/w, h/h, w/w


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class SingleRandomResizedCrop(transforms.RandomResizedCrop):
    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, width

    def rescale_boxes(self, boxes, i, j, old_h, old_w):
        """Rescale bounding boxes according to the crop parameters"""

        _boxes = boxes.clone()
        _boxes[:, 0::2] -= j
        _boxes[:, 1::2] -= i
        for i in range(_boxes.shape[0]):
            if _boxes[i, 0] < 0 or _boxes[i, 1] < 0:
                _boxes[i, :] = -1
            if _boxes[i, 2] >= old_w or _boxes[i, 3] >= old_h:
                _boxes[i, :] = -1
        if isinstance(self.size, tuple):
            ratio_w = self.size[1] / old_w
            ratio_h = self.size[0] / old_h
        else:
            ratio_w = self.size / old_w
            ratio_h = self.size / old_h
        _boxes = _boxes.float()

        mask = torch.all(_boxes != -1, dim=1)

        _boxes[mask, 0::2] *= ratio_w
        _boxes[mask, 1::2] *= ratio_h

        return _boxes.int()

    def forward(self, img, boxes):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w, width = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
               self.rescale_boxes(boxes, i, j, h, w), i, j, h, w, width


class Resize(transforms.Resize):
    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        super(Resize, self).__init__(size, interpolation)

    def rescale_boxes(self, boxes, old_h, old_w):
        _boxes = boxes.clone()

        if isinstance(self.size, tuple):
            ratio_w = self.size[1] / old_w
            ratio_h = self.size[0] / old_h
        else:
            ratio_w = self.size / old_w
            ratio_h = self.size / old_h

        _boxes = boxes.float()

        mask = torch.all(_boxes != -1, dim=1)

        _boxes[mask, 0::2] *= ratio_w
        _boxes[mask, 1::2] *= ratio_h

        return _boxes.int()

    def forward(self, img, boxes):
        w, h = F.get_image_size(img)

        img = transforms.Resize(size=self.size, interpolation=self.interpolation)(img)
        boxes = self.rescale_boxes(boxes, h, w)

        return img, boxes


class RandomHorizontalFlipForMaps(transforms.RandomHorizontalFlip):
    """Wrapper around imgaug flip for multiple masks"""
    def __init__(self, train=True):
        super().__init__()

        if train:
            p = 1.0
        else:
            p = 0.0

        self.aug = iaa.Fliplr(p)

    def forward(self, img, type_map, inst_map):
        if torch.rand(1) < self.p:

            aug_img, type_map = self.aug(image=img, segmentation_maps=type_map)
            _, inst_map = self.aug(image=img, segmentation_maps=inst_map)
            return aug_img, type_map, inst_map
        return img, type_map, inst_map


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def flip_boxes(self, boxes, width):
        _boxes = boxes.clone()
        mask = torch.all(_boxes != -1, dim=1)

        w = _boxes[mask, 2] - _boxes[mask, 0]

        _boxes[mask, 0::2] *= -1
        _boxes[mask, 0::2] += width - 1

        _boxes[mask, 0] -= w
        _boxes[mask, 2] += w
        return _boxes

    def forward(self, img, boxes):
        if torch.rand(1) < self.p:
            w, _ = F.get_image_size(img)
            return F.hflip(img), True, self.flip_boxes(boxes, w)
        return img, False, boxes


def _pad_boxes_and_classes(boxes, classes, num_total):
    fake_box = torch.tensor(4 * [-1])
    if boxes is None:
        if classes is None:
            return fake_box.expand(num_total, -1), None
        else:
            return fake_box.expand(num_total, -1), fake_class.expand(num_total, -1)

    items_available = boxes.shape[0]

    if boxes.shape[0] <= num_total:
        padding_length = num_total - items_available
        fake_box = fake_box.expand(padding_length, -1)
        fake_class = torch.tensor(padding_length * [-1])
        boxes = torch.cat([boxes, fake_box])
        if classes is None:
            return boxes, None
        else:
            classes = torch.cat([classes, fake_class])
            return boxes, classes

    idx = random.sample(range(items_available), num_total)

    if classes is None:
        return boxes[idx], None
    else:
        return boxes[idx], classes[idx]

