import time
import os

import torch, gc
import numpy as np
import torch.distributed as dist
from metrices import Summary, AverageMeter, ProgressMeter, EvaluationMatrices


class Trainer:
    def __init__(self, model, train_data, val_data, criterion, optimizer, early_stop=5, args=None, device=None):
        self._model = model
        self._train_data = train_data
        self._val_data = val_data
        self._crit = criterion
        self._optim = optimizer
        self._early_stop = early_stop
        self.args = args

        self._counter = 0
        self.best_loss = np.inf

        self.accuracy = EvaluationMatrices()

        # setup
        self.weight_path = f'./checkpoints/{args.model}_weights.pkl'
        self.loss_path = f'./checkpoints/{args.model}_loss.pkl'
        self.acc1_path = f'./checkpoints/{args.model}_acc1.pkl'
        self.acc5_path = f'./checkpoints/{args.model}_acc5.pkl'

    def _training_epochs(self):

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        train_loss = AverageMeter('Train Loss', ':.4e', Summary.NONE)
        top1_train = AverageMeter('Train Acc@1', ':6.2f', Summary.AVERAGE)
        top5_train = AverageMeter('Train Acc@5', ':6.2f', Summary.AVERAGE)

        train_progress = ProgressMeter(
            len(self._train_data),
            [batch_time, train_loss, top1_train, top5_train],
            prefix="Training: ")

        self._model.train()
        end = time.time()

        for batch_id, (image, label) in enumerate(self._train_data):
            if self.args.gpu is not None and torch.cuda.is_available():
                image = image.cuda(self.args.local_rank, non_blocking=True)
                label = label.cuda(self.args.local_rank, non_blocking=True)

            # training step
            self._optim.zero_grad()
            training_pred = self._model(image)
            loss = self._crit(training_pred, label)
            loss.backward()
            self._optim.step()

            # compute accuracies and losses
            acc1, acc5 = self.accuracy.top_accuracy(training_pred, label, len(label), topk=(1, 5))
            train_loss.update(loss.item(), image.size(0))
            top1_train.update(acc1[0], image.size(0))
            top5_train.update(acc5[0], image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_id % self.args.print_freq == 0:
                print(f'batch_id: {batch_id}')
                train_progress.display(batch_id)

        train_progress.display_summary()

        return top1_train.avg, top5_train.avg, train_loss.avg

    def _validation_epochs(self):

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        val_loss = AverageMeter('Validation Loss', ':.4e', Summary.NONE)
        top1_val = AverageMeter('Validation Acc@1', ':6.2f', Summary.AVERAGE)
        top5_val = AverageMeter('Validation Acc@5', ':6.2f', Summary.AVERAGE)

        val_progress = ProgressMeter(
            len(self._train_data),
            [batch_time, val_loss, top1_val, top5_val],
            prefix="Validation: ")

        self._model.eval()

        with torch.no_grad():
            end = time.time()
            for batch_id, (image, label) in enumerate(self._val_data):
                if self.args.gpu is not None and torch.cuda.is_available():
                    image = image.cuda(self.args.local_rank, non_blocking=True)
                    label = label.cuda(self.args.local_rank, non_blocking=True)

                # validation step
                val_pred = self._model(image)
                loss = self._crit(val_pred, label)

                # compute accuracies and losses
                acc1, acc5 = self.accuracy.top_accuracy(val_pred, label, len(label), topk=(1, 5))
                val_loss.update(loss.item(), image.size(0))
                top1_val.update(acc1[0], image.size(0))
                top5_val.update(acc5[0], image.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_id % self.args.print_freq == 0:
                    val_progress.display(batch_id)

            val_progress.display_summary()

        return top1_val.avg, top5_val.avg, val_loss.avg


    def fit(self, epochs=25):
        train_loss, val_loss = (list(), list()) if not self.args.checkpoints else torch.load(self.args.checkpoint_path + f'{self.args.model}_loss.pkl', map_location=torch.device("cuda", self.args.local_rank))
        train_acc1, val_acc1 = (list(), list()) if not self.args.checkpoints else torch.load(self.args.checkpoint_path + f'{self.args.model}_acc1.pkl', map_location=torch.device("cuda", self.args.local_rank))
        train_acc5, val_acc5 = (list(), list()) if not self.args.checkpoints else torch.load(self.args.checkpoint_path + f'{self.args.model}_acc5.pkl', map_location=torch.device("cuda", self.args.local_rank))

        for epoch in range(epochs):
            if not self.args.train_art:
                self._train_data.sampler.set_epoch(epoch)

            training = self._training_epochs()
            train_acc1.append(training[0])
            train_acc5.append(training[1])
            train_loss.append(training[2])

            validation = self._validation_epochs()
            val_acc1.append(validation[0])
            val_acc5.append(validation[1])
            val_loss.append(validation[2])

            # check for early stopping
            if validation[2] < self.best_loss:
                self.best_loss = validation[2]
                torch.save(self._model.state_dict(), self.weight_path)

            gc.collect()
            torch.cuda.empty_cache()

        # save the model
        torch.save([train_loss, val_loss], self.loss_path)
        torch.save([train_acc1, val_acc1], self.acc1_path)
        torch.save([train_acc5, val_acc5], self.acc5_path)

        return (train_acc1, val_acc1), (train_acc5, val_acc5), (train_loss, val_loss)






