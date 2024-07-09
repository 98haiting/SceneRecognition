import time

import torch
import numpy as np
import math
from metrices import Summary, AverageMeter, ProgressMeter, EvaluationMatrices

class Tester:
    def __init__(self, model, test_data, criterion, args=None):
        self._model = model
        self._test_data = test_data
        self._crit = criterion
        self.args = args

        self.accuracy = EvaluationMatrices()

        # setup
        self.acc_path = f'./checkpoints/{args.model}_test_acc.pkl'
        self.quantative_results = f"./checkpoints/{args.result_config}_quantitative_results.pkl"

    def predictor(self):
        label_counter = 0
        top1_acc = AverageMeter('Test Acc@1', ':6.2f', Summary.AVERAGE)
        top5_acc = AverageMeter('Test Acc@5', ':6.2f', Summary.AVERAGE)
        f1_score = AverageMeter('F1', ':6.2f', Summary.AVERAGE)

        self._model.eval()
        ground_truth, prediction = [], []

        with torch.no_grad():
            for batch_id, (image, label) in enumerate(self._test_data):
                image = image.cuda(self.args.local_rank, non_blocking=True)
                label = label.cuda(self.args.local_rank, non_blocking=True)
                # validation step
                val_pred = self._model(image)

                ground_truth.append(label)
                prediction.append(val_pred)

                if not math.isnan(label):
                    label_counter += 1
                    acc1, acc5 = self.accuracy.top_accuracy(val_pred, label, len(label), topk=(1, 5))
                    f1 = self.accuracy.F1_score(val_pred, label)
                    top1_acc.update(acc1[0], image.size(0))
                    top5_acc.update(acc5[0], image.size(0))
                    f1_score.update(f1, image.size(0))

        torch.save([top1_acc.avg, top5_acc.avg, f1_score.avg], self.acc_path)
        torch.save([ground_truth, prediction], self.quantative_results)

        print(f"top1 accuracy: {top1_acc.avg}, top5 accuracy: {top5_acc.avg}, f1-score: {f1_score.avg}")

        return top1_acc.avg, top5_acc.avg, f1_score.avg



