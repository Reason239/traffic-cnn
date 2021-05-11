from dataset import MyDataset
from model import small_resnet14
from training import Trainer
from dataset import TRAFFIC_LABELS
import pathlib
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import mode
import torch

name = 'small_14_adam_1em3_bs128_'
base_path = pathlib.Path('models') / name
state_dict_path = base_path / 'best_model.pth'

model = small_resnet14()
model.load_state_dict(torch.load(state_dict_path))

batch_size = 256

trainer = Trainer(model, None, None, None, None, None, 'cuda', batch_size=batch_size, save_path=base_path)

data_path_root_test = pathlib.Path('test/')
test_anno = pd.DataFrame({'id': [f'pic{num:06}' for num in range(10699)],
                          'category': [0 for num in range(10699)]})
test_dataset = MyDataset(data_dir=data_path_root_test, data_anno=test_anno, phase='train')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
all_preds_list = []
for i in range(7):
    all_preds_list.append(trainer.predict(test_dataloader, labels=True).flatten())
print((all_preds_list[0] != all_preds_list[1]).mean())
val, count = mode(all_preds_list, axis=0)
preds = val.ravel().tolist()
submit = pd.DataFrame({'id': [f'pic{num:06}' for num in range(10699)],
                       'category': [TRAFFIC_LABELS[pred] for pred in preds]})


submit.to_csv(base_path / 'submit_tta.csv', index=False)
print('Done')
