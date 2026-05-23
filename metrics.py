import torch
from torchmetrics.functional.classification import multiclass_accuracy

def calculate_topk_accuracy(output, target, topk=(1, 5)):

    res = []
    for k in topk:
        acc = multiclass_accuracy(output, target, num_classes=37, top_k=k)
        res.append(acc.item() * 100.0)
    return res



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
