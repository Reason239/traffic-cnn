import cv2
import torch

image = cv2.imread('image.png')
print(image)
print(image.shape, image.dtype)

image = torch.from_numpy(image).float()
print(image.shape, image.dtype)
image = image.permute(2, 0, 1)
print(image.shape)
image = (image - 127.5) / 128
print(image)
