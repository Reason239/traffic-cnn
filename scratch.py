# # import torch
# # import torch.nn as nn
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from dataset import TRAFFIC_LABELS_TO_NUM
#
# data = pd.read_csv('train_val/keys.csv')[:3]
# # print(len(data))
# # print(data['category'].map(TRAFFIC_LABELS_TO_NUM))
# # print(set(data['category'].unique()))
# # print(set(TRAFFIC_LABELS_TO_NUM.keys()))
# print(data)
# data.loc['category'] = data.loc['category'].map(TRAFFIC_LABELS_TO_NUM)
# print(data)
# # print(data.columns)
# # print(data.head())
# # print(data.info())
# # is_NaN = data.isnull()
# # row_has_NaN = is_NaN.any(axis=1)
# # rows_with_NaN = data[row_has_NaN]
# # print(rows_with_NaN)
# # print(data.isnull().any())
# print(data['category'])
# print(data['category'].map(TRAFFIC_LABELS_TO_NUM))
# # print(data)
# train, test = train_test_split(data, test_size=len(data)//5, stratify=data['category'])
# # print(train.head())
import numpy as np
from scipy.stats import mode

a = np.array([0, 0, 2])
b = np.array([2, 1, 2])
c = np.array([0, 1, 3])
l = [a, b, c]
val, count = mode(l, axis=0)
print(val.ravel().tolist())