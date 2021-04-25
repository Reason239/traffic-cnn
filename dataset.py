import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import albumentations as alb
import albumentations.augmentations.transforms as aat

FIXED_IMG_HEIGHT = 64
FIXED_IMG_WIDTH = 64

TRAFFIC_LABELS = ('disabled', 'green', 'red', 'red_yellow', 'yellow')
TRAFFIC_LABELS_TO_NUM = {label: num for num, label in enumerate(TRAFFIC_LABELS)}


def resize_img(img, res_shape=(FIXED_IMG_HEIGHT, FIXED_IMG_WIDTH)):
    height, width = img.shape[:2]
    resized_width = int(res_shape[0] * (float(width) / height))
    img_resized = cv2.resize(img, (resized_width, int(res_shape[0])))
    width_to_copy = min(resized_width, res_shape[1])
    img_resized_filled_with_zero = np.zeros([res_shape[0], res_shape[1], 3], dtype=img.dtype)
    img_resized_filled_with_zero[:, :width_to_copy, :] = img_resized[:, :width_to_copy, :]
    return img_resized_filled_with_zero


def preprocess(img):
    img = img.transpose(2, 0, 1)
    img = (img - 127.5) / 128
    img = np.array(img, dtype=np.float32)

    return img


class AlbuWrapper:
    def __init__(self, atrans):
        self.atrans = atrans

    def __call__(self, img):
        return self.atrans(image=np.array(img))["image"]


alb_transforms = AlbuWrapper(alb.Compose(
    [
        alb.OneOf([alb.IAAAdditiveGaussianNoise(), alb.GaussNoise()], p=0.2),
        alb.OneOf([alb.MotionBlur(p=0.2), alb.MedianBlur(blur_limit=3, p=0.1),
                   alb.Blur(blur_limit=3, p=0.1), alb.RandomFog(p=0.1)], p=0.2),
        alb.OneOf([alb.OpticalDistortion(p=0.3), alb.GridDistortion(p=0.1),
                   alb.IAAAffine(p=0.1), alb.ShiftScaleRotate(rotate_limit=15, p=0.1)], p=0.2),
        alb.OneOf([alb.CLAHE(clip_limit=2), alb.IAASharpen(), alb.IAAEmboss()], p=0.3),
        alb.OneOf([alb.HueSaturationValue(p=0.3), alb.RGBShift(),
                   alb.RandomBrightnessContrast(), alb.Equalize(mode='cv', p=1)], p=0.3),
        alb.HorizontalFlip(),
        alb.RandomGamma(p=0.2),
        alb.OneOf([alb.CoarseDropout(2, 10, 10, p=0.5), alb.RandomRain(blur_value=3, p=1)], p=0.5)
    ]))


class MyDataset(Dataset):
    def __init__(self, data_dir, data_anno, phase='train', transform=alb_transforms, max_size=None):
        self.data_dir = data_dir
        self.data_anno = data_anno
        self.phase = phase
        self.transform = transform
        if max_size is not None:
            self.data_anno = self.data_anno[:max_size]

    def __getitem__(self, idx):
        pic_id = self.data_anno.iloc[idx]['id']
        # print(os.path.join(self.data_dir / 'pic' / f'{pic_id}.jpg'))
        image = cv2.imread(os.path.join(self.data_dir / 'pic' / f'{pic_id}.jpg'))
        # print(self.data_dir / f'{pic_id}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.phase == 'test':
            image = resize_img(image)
            image = preprocess(image)
            return torch.from_numpy(image).float()

        label = self.data_anno.iloc[idx]['category']

        if self.phase == 'train' and self.transform is not None:
            image = self.transform(image)

        image = resize_img(image)
        image = preprocess(image)

        return torch.from_numpy(image).float(), torch.from_numpy(np.array(label)).to(torch.long)

    def __len__(self):
        return len(self.data_anno)


def tensor2img(tensor):
    return (tensor * 128 + 127.5).numpy().astype(np.uint8).transpose(1, 2, 0)
