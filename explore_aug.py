from dataset import MyDataset, AlbuWrapper, tensor2img
import albumentations as alb
import albumentations.augmentations.transforms as aat
import albumentations
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pathlib
from dataset import TRAFFIC_LABELS_TO_NUM

alb_transforms = [
    alb.IAAAdditiveGaussianNoise(p=1),
    alb.GaussNoise(p=1),
    alb.MotionBlur(p=1),
    alb.MedianBlur(blur_limit=3, p=1),
    alb.Blur(blur_limit=3, p=1),
    alb.OpticalDistortion(p=1),
    alb.GridDistortion(p=1),
    alb.IAAPiecewiseAffine(p=1),
    aat.CLAHE(clip_limit=2, p=1),
    alb.IAASharpen(p=1),
    alb.IAAEmboss(p=1),
    aat.HueSaturationValue(p=0.3),
    aat.HorizontalFlip(p=1),
    aat.RGBShift(),
    aat.RandomBrightnessContrast(),
    aat.RandomGamma(p=1),
    aat.Cutout(2, 10, 10, p=1),
    aat.Equalize(mode='cv', p=1),
    aat.FancyPCA(p=1),
    aat.RandomFog(p=1),
    aat.RandomRain(blur_value=3, p=1),
    albumentations.IAAAffine(p=1),
    albumentations.ShiftScaleRotate(rotate_limit=15, p=1)
]

def one_by_one():
    data_dir = 'train_val/pic'
    data_anno = pd.read_csv('train_val/keys.csv')
    orig = MyDataset(data_dir, data_anno)
    for transform in tqdm(alb_transforms):
        str_transform = str(transform)
        str_transform = str_transform[:str_transform.find('(')]
        path = f'aug_pics/{str_transform}.png'
        if not pathlib.Path(path).exists():
            data = MyDataset(data_dir, data_anno, transform=AlbuWrapper(transform))
            fig, axes = plt.subplots(4, 2, figsize=(4, 8))
            for row in axes:
                for ax in row:
                    ax.set_axis_off()
            for i in range(4):
                tens, _ = orig.__getitem__(i)
                img = tensor2img(tens)
                tens, _ = data.__getitem__(i)
                img_aug = tensor2img(tens)
                axes[i, 0].imshow(img)
                axes[i, 1].imshow(img_aug)
            fig.suptitle(str_transform)
            fig.savefig(path)

def composition():
    data_dir = pathlib.Path('train_val')
    data_anno_raw = pd.read_csv('train_val/keys.csv')
    data_anno = pd.DataFrame({'id': data_anno_raw['id'].values,
                              'category': [TRAFFIC_LABELS_TO_NUM[label] for label in data_anno_raw['category'].values]})
    orig = MyDataset(data_dir, data_anno, transform=None)
    str_transform = "Composition"
    path = f'aug_pics/{str_transform}.png'
    data = MyDataset(data_dir, data_anno)
    fig, axes = plt.subplots(8, 2, figsize=(4, 16))
    for row in axes:
        for ax in row:
            ax.set_axis_off()
    for i in range(8):
        tens, _ = orig.__getitem__(i)
        img = tensor2img(tens)
        tens, _ = data.__getitem__(i)
        img_aug = tensor2img(tens)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(img_aug)
    fig.suptitle(str_transform)
    fig.savefig(path)

if __name__ == '__main__':
    composition()