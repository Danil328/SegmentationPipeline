import glob
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Compose,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    OneOf,
    RandomBrightnessContrast,
    CLAHE,
    Normalize,
    ShiftScaleRotate,
    CropNonEmptyMaskIfExists, RandomResizedCrop, Resize, ImageCompression, RandomGamma, RandomCrop, Rotate)
from albumentations.pytorch import ToTensor, ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm


def make_aug(num_classes, shape, phase):
    if phase == 'train':
        aug = Compose([
                Resize(height=shape[0], width=shape[1], always_apply=True),
                RandomRotate90(p=0.25),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=(-0.2, 0.2), scale_limit=(-0.2, 0.2), rotate_limit=(-20, 20), border_mode=0, interpolation=1, p=0.25),
                OneOf([
                    ElasticTransform(p=0.2, alpha=120, sigma=120 * 0.1, alpha_affine=120 * 0.03),
                    GridDistortion(p=0.5),
                    OpticalDistortion(border_mode=0, distort_limit=0.05, interpolation=1, shift_limit=0.05, p=1.0),
                ], p=0.2),
                ImageCompression(quality_lower=80, p=0.5),
                Normalize(),
                ToTensor(num_classes=num_classes, sigmoid=True)
            ], p=1)
    else:
        aug = Compose([
                Resize(height=shape[0], width=shape[1], always_apply=True),
                Normalize(),
                ToTensor(num_classes=num_classes, sigmoid=True)
            ], p=1)
    return aug


class SegmentationDataset(Dataset):
    def __init__(self, data_folder, phase, num_classes, shape, fold=-1, fp16=False):
        assert phase in ['train', 'val', 'test', 'holdout'], "Fuck you!"

        self.root = data_folder
        self.num_classes = num_classes
        self.transforms = make_aug(num_classes, shape, phase)
        self.shape = shape
        self.phase = phase
        self.fold = fold
        self.fp16 = fp16

        self.images = np.asarray(self.split_train_val(glob.glob(os.path.join(self.root, "*.png"))))
        self.masks = np.asarray(list(map(lambda x: x, self.images)))

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])

        if self.phase != 'test':
            mask = cv2.imread(self.images[idx])
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

            if self.fp16:
                return {"image": img.half(), "mask": mask.half(), "filename": self.images[idx].split("/")[-1]}
            else:
                return {"image": img, "mask": mask, "filename": self.images[idx].split("/")[-1]}
        else:
            augmented = self.transforms(image=img)
            img = augmented['image']
            if self.fp16:
                return {"image": img.half(), "filename": self.images[idx].split("/")[-1]}
            else:
                return {"image": img, "filename": self.images[idx].split("/")[-1]}

    def __len__(self):
        return len(self.images)

    def split_train_val(self, images: list):
        if self.fold < 0:
            train, val = train_test_split(images, test_size=0.1, random_state=17)
            if self.phase == 'train':
                return train
            elif self.phase == 'val':
                return val
        else:
            cv_df = pd.read_csv(os.path.join(self.root, '..', f'cross_val_DF_{self.orig_shape[0]}.csv'))
            return cv_df[cv_df[f'fold_{self.fold}'] == self.phase]['images'].values


if __name__ == '__main__':
    rnd = np.random.randint(1, 60)
    dataset = SegmentationDataset(data_folder=os.path.join('/home/danil.akhmetov/Projects/SeaIce/data/', "train_images"),
                                  phase='train', fold=1, num_classes=6, shape=(512, 512))
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True)

    data = dataset[rnd]
    image = data['image'].numpy()
    mask = data['mask'].numpy()
    filename = data['filename']

    print(filename)
    print(image.shape)
    print(mask.shape)

    print(image.min(), image.max())
    print(mask.min(), mask.max())
