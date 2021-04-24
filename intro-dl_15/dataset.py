import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def preprocess(img):
    img = img.transpose(2, 0, 1)
    img = (img - 127.5) / 128
    img = np.array(img, dtype=np.float32)

    return img


class MyDataset(Dataset):

    def __init__(self, data_anno, phase='train', transform=None):
        self.data_anno = data_anno
        self.transform = transform
        self.phase = phase

    def __getitem__(self, idx):

        if self.phase == 'test':
            image_file = self.data_anno.iloc[idx]['Id']
            image = cv2.imread('/home/iris/data_for_introdl/data/test/' + image_file)
            image = cv2.resize(image, (224, 224))
            image = preprocess(image)

            return torch.from_numpy(image).float()

        image_file = self.data_anno.iloc[idx]['Id']
        label = self.data_anno.iloc[idx]['Category']

        image = cv2.imread('/home/iris/data_for_introdl/data/train/' + image_file)
        image = cv2.resize(image, (224, 224))

        if self.phase == 'train' and self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        image = preprocess(image)

        return torch.from_numpy(image).float(), label

    def __len__(self):
        return len(self.data_anno)
