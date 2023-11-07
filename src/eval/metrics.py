import torch
import torchmetrics
import torch.nn as nn


class Metrics:
    def __init__(self, device):
        self.mse = nn.MSELoss()
        self.jcd = torchmetrics.JaccardIndex(task="multiclass", num_classes=2).to(
            device
        )
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)

    def mean_squared_loss(self, pred, true):
        return self.mse(pred, true)

    def jaccard_score(self, pred, true):
        return self.jcd(pred, true)

    def accuracy_score(self, pred, true):
        return self.acc(pred, true)
