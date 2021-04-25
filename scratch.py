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

import comet_ml
COMET_API_KEY = 'oh07wzIdnbJ3Obu4mEDzNT9MF'
comet_experiment = comet_ml.Experiment(api_key=COMET_API_KEY, project_name='Traffic CNN',
                                       log_git_patch=False, log_git_metadata=False,
                                       auto_output_logging=False,
                                       auto_histogram_weight_logging=False, auto_histogram_gradient_logging=False)
print('Done')