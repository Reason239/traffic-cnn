import comet_ml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from dataset import MyDataset, tensor2img, TRAFFIC_LABELS, TRAFFIC_LABELS_TO_NUM
import pandas as pd
import matplotlib.pyplot as plt
from training import Trainer, find_lr
from dataset import MyDataset, FIXED_IMG_HEIGHT
from model import small_resnet8, small_resnet14, test_model
from sklearn.model_selection import train_test_split
import pathlib


if __name__ == '__main__':
    # Hyperparameters
    name = 'small_14_adam_1em3_cycle_20'
    max_lr = 1e-3
    min_lr = 1e-4
    device = 'cuda'
    num_epochs = 20
    batch_size = 128
    batches_per_epoch = -1
    hyperparameters = dict(max_lr=max_lr, min_lr=min_lr, device=device, num_epochs=num_epochs, batch_size=batch_size)

    # Comet
    comet_experiment = comet_ml.Experiment(api_key=COMET_API_KEY, project_name='Traffic CNN',
                                           auto_output_logging=False, log_git_patch=False, log_git_metadata=False,
                                           auto_histogram_weight_logging=True, auto_histogram_gradient_logging=True)
    comet_experiment.log_parameters(hyperparameters)
    comet_experiment.set_name(name)
    # comet_experiment = None

    # Model
    model = small_resnet14()
    print(f'Number of parameters in model: ',
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Data
    data_path_root = pathlib.Path('train_val/')
    data_anno_raw = pd.read_csv(data_path_root / 'keys.csv')
    data_anno = pd.DataFrame({'id': data_anno_raw['id'].values,
                              'category': [TRAFFIC_LABELS_TO_NUM[label] for label in data_anno_raw['category'].values]})
    data_size = len(data_anno)
    test_size = data_size // 5
    train_anno, val_anno = train_test_split(data_anno, test_size=test_size, stratify=data_anno['category'],
                                            random_state=42)
    train_dataset = MyDataset(data_dir=data_path_root, data_anno=train_anno, phase='train')
    val_dataset = MyDataset(data_dir=data_path_root, data_anno=val_anno, phase='eval')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Criterion (add weights?)
    criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.5 * 1e-3, steps_per_epoch=len(train_dataloader),
                                                    epochs=num_epochs)

    # Training
    # optim_gen = lambda parameters, lr: SGD(parameters, lr=lr)
    # find_lr(model, optim_gen, min_lr, max_lr, num_epochs, train_dataloader, val_dataloader, criterion, device, batch_size,
    #         batches_per_epoch, comet_experiment)
    save_path = pathlib.Path('models') / name
    save_path.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, None, device, TRAFFIC_LABELS,
                      num_epochs, batch_size, batches_per_epoch, comet_experiment, save_path)
    try:
        trainer.fit()
    except KeyboardInterrupt:
        pass

    # Prediction
    data_path_root_test = pathlib.Path('test/')
    test_anno = pd.DataFrame({'id': [f'pic{num:06}' for num in range(10699)]})
    test_dataset = MyDataset(data_dir=data_path_root_test, data_anno=test_anno, phase='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = trainer.predict(test_dataloader)
    submit = pd.DataFrame({'id': [f'pic{num:06}' for num in range(10699)],
                           'category': [TRAFFIC_LABELS[pred] for pred in preds]})
    submit.to_csv(save_path / 'submit.csv')

    print('Done')
