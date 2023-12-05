import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from PIL import Image

from utils import read_img
from constants import IMAGE_SIZE, DATA_DIR, PROCESSED_DICOM_DIR


def individual_normalize(image, **kwargs):
    img_mean = image.mean()
    img_std = image.std()
    return (image - img_mean) / img_std


def custom_scale(image, **kwargs):
    min_value = image.min()
    max_value = image.max()
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def equalize_image(image, **kwargs):
    min_value = image.min()
    max_value = image.max()
    image = ((image - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)
    image = Image.fromarray(image)
    return np.array(transforms.functional.equalize(image))


class ResizeNoWarp(A.DualTransform):
    def __init__(self, target_size=2048, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.target_size = target_size
    
    def apply(self, image, **params):
        current_height, current_width = image.shape[:2]

        # Calculate the scaling factors for resizing
        scale_x = self.target_size / current_width
        scale_y = self.target_size / current_height

        # Calculate the scaling factor for the smaller dimension
        scaling_factor = min(scale_x, scale_y)

        # If the image is already smaler than the target size, don't resize
        if scaling_factor >= 1:
            return image

        new_height, new_width = self.new_size(image.shape[:2])
        image = cv2.resize(image, (new_width, new_height))
        return image

    def new_size(self, image_shape):
        aspect_ratio = image_shape[1] / image_shape[0]
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        return new_height, new_width


class RandomPad(A.DualTransform):
    def __init__(self, target_size=2048, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.target_size = target_size
    
    def apply(self, image, **params):
        pad_top, pad_bottom, pad_left, pad_right = self.random_padding(image.shape[:2])
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return image

    def random_padding(self, image_shape):
        height, width = image_shape[0], image_shape[1]
        pad_height_total = self.target_size - height
        pad_width_total = self.target_size - width

        pad_top = random.randint(0, pad_height_total)
        pad_bottom = pad_height_total - pad_top
        pad_left = random.randint(0, pad_width_total)
        pad_right = pad_width_total - pad_left
    
        return pad_top, pad_bottom, pad_left, pad_right    


class CenterPad(A.DualTransform):
    def __init__(self, target_size=2048, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.target_size = target_size

    def apply(self, image, **params):
        pad_top, pad_bottom, pad_left, pad_right = self.random_padding(image.shape[:2])
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return image

    def random_padding(self, image_shape):
        height, width = image_shape[0], image_shape[1]
        pad_height_total = self.target_size - height
        pad_width_total = self.target_size - width

        pad_top = pad_height_total // 2
        pad_bottom = pad_height_total - pad_top
        pad_left = pad_width_total // 2
        pad_right = pad_width_total - pad_left

        return pad_top, pad_bottom, pad_left, pad_right


class RandomResizeAndPad(A.DualTransform):
    def __init__(self, target_size=512, always_apply=True, p=1):
        super().__init__(always_apply, p)
        self.target_size = target_size

    def apply(self, image, **params):
        new_height, new_width, pad_top, pad_bottom, pad_left, pad_right = self.random_padding(image.shape[:2])
        image = cv2.resize(image, (new_width, new_height))
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        return image

    def random_padding(self, image_shape):
        aspect_ratio = image_shape[1] / image_shape[0]
        if aspect_ratio > 1:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)

        pad_height_total = self.target_size - new_height
        pad_width_total = self.target_size - new_width

        pad_top = random.randint(0, pad_height_total)
        pad_bottom = pad_height_total - pad_top
        pad_left = random.randint(0, pad_width_total)
        pad_right = pad_width_total - pad_left

        return new_height, new_width, pad_top, pad_bottom, pad_left, pad_right


train_transform = A.Compose([
    ResizeNoWarp(target_size=IMAGE_SIZE),
    A.Lambda(image=equalize_image),
    A.Lambda(image=custom_scale),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit=(-.05,.05),
                       shift_limit=(-.05,.05),
                       rotate_limit=(0,0), 
                       border_mode=cv2.BORDER_CONSTANT,
                       value=0,
                       p=0.5),
    A.Rotate(limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
    ], p=0.3),
    RandomPad(target_size=IMAGE_SIZE),
    A.Lambda(image=individual_normalize),
    ToTensorV2()
])

validation_transform = A.Compose([
    ResizeNoWarp(target_size=IMAGE_SIZE),
    A.Lambda(image=equalize_image),
    A.Lambda(image=custom_scale),
    CenterPad(target_size=IMAGE_SIZE),
    A.Lambda(image=individual_normalize),
    ToTensorV2()
])

test_transform = A.Compose([
    ResizeNoWarp(target_size=IMAGE_SIZE),
    A.Lambda(image=equalize_image),
    A.Lambda(image=custom_scale),
    CenterPad(target_size=IMAGE_SIZE),
    A.Lambda(image=individual_normalize),
    ToTensorV2()
])


class CustomDataset(Dataset):
    def __init__(self, which, augmentations=None, sample_size=None, kind='PORT CHEST'):
        self.which = which
        self.df = pd.read_csv(DATA_DIR + '/' + which + '_data.csv')
        
        if sample_size is not None and self.which == 'train':
            self.df = self.df.sample(sample_size).reset_index(drop=True)
        if kind is not None:
            self.df = self.df[self.df['Body Part Examined'] == kind].reset_index(drop=True)

        self.paths = self.df['Path']
        self.labels = self.df['Label']
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            path = self.paths[idx]
            img = read_img(path)
            if img is None:
                print(f"Warning: Image is None for index {idx}. Path: {path}")

            label = self.labels[idx].astype(int)
            # Apply augmentations
            if self.augmentations:
                augmented = self.augmentations(image=img)
                img = augmented['image']

            return img, label, path
        except Exception as e:
            print("Error:", e)
