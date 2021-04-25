import torch
from torch.optim.lr_scheduler import ExponentialLR
from comet_ml import Experiment
from copy import deepcopy
import comet_ml
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.auto import trange
from dataset import TRAFFIC_LABELS


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device,
                 labels=TRAFFIC_LABELS, num_epochs=20, batch_size=20, batches_per_epoch=-1,
                 comet_experiment: comet_ml.Experiment = None, save_path=None):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.labels = labels
        self.num_classes = len(labels)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.num_epochs = num_epochs
        self.comet_experiment = comet_experiment
        self.save_path = save_path

        self.best_val_acc = -1

    def epoch_loop(self, phase):
        if phase == 'train':
            self.model.train()
            dataloader = self.train_dataloader
        elif phase == 'eval':
            self.model.eval()
            dataloader = self.val_dataloader
        else:
            raise ValueError('Incorrect phase value')

        loss = 0
        correct = 0
        confusion_matr = np.zeros((self.num_classes, self.num_classes), dtype=int)
        num_batches = 0
        num_images = 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            batch_loss = self.criterion(output, labels)
            if phase == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            batch_loss_float = batch_loss.cpu().item()
            loss += batch_loss_float
            if self.comet_experiment:
                self.comet_experiment.log_metric('batch loss', batch_loss_float)
            _, preds = torch.max(output.data, 1)
            labels, preds = labels.cpu().numpy(), preds.cpu().numpy()
            correct += (labels == preds).sum()
            confusion_matr += confusion_matrix(labels, preds, labels=list(range(self.num_classes)))
            num_batches += 1
            num_images += len(images)
            if num_batches == self.batches_per_epoch:
                break

        loss /= num_batches
        accuracy = correct / num_images
        # confusion_matr = confusion_matr / num_images

        if self.comet_experiment:
            metrics = {f'{phase}_loss': loss,
                       f'{phase}_laccuracy': accuracy,
                       }
            self.comet_experiment.log_metrics(metrics)
            self.comet_experiment.log_confusion_matrix(matrix=confusion_matr, labels=self.labels)

        if phase == 'eval' and accuracy > self.best_val_acc:
            self.best_val_acc = accuracy
            if self.save_path:
                torch.save(self.model.state_dict(), self.save_path / 'best_model.pth')

        return loss, accuracy

    def fit(self):
        """
        Trains a model
        """
        # Train loop (add gradient logging?)
        # Val loop (add confusion matrix logging?)
        # Scheduler update
        # Comet logging and model snapshoting

        print('Begin training')

        self.best_val_acc = -1
        self.criterion = self.criterion.to(self.device)
        self.model = self.model.to(self.device)

        tqdm_range = trange(self.num_epochs)
        for _ in tqdm_range:
            train_loss, train_acc = self.epoch_loop(phase='train')
            val_loss, val_acc = self.epoch_loop(phase='eval')
            if self.scheduler:
                self.scheduler.step()
            tqdm_range.set_postfix(train_loss=train_loss, val_loss=val_loss,
                                   train_acc=train_acc, val_acc=val_acc)
        print('Finish training')


def find_lr(model, optimizer_gen, min_lr, max_lr, num_epochs, train_dataloader, val_dataloader, criterion, device,
            batch_size, batches_per_epoch, comet_experiment):
    optimizer = optimizer_gen(model.parameters(), lr=max_lr)
    # max_lr * factor ** num_epochs = min_lr
    factor = (np.log(min_lr) - np.log(max_lr)) / num_epochs
    scheduler = ExponentialLR(optimizer, gamma=1 - factor, verbose=True)
    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, device, TRAFFIC_LABELS,
                      num_epochs, batch_size, batches_per_epoch, comet_experiment, None)
    trainer.fit()
