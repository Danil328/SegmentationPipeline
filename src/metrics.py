import numpy as np
import torch
from sklearn.metrics import accuracy_score, jaccard_score
from torch import nn


def dice_coef(preds, trues, smooth=1e-3):
    preds = preds.contiguous().view(preds.size(0), -1).float()
    trues = trues.contiguous().view(preds.size(0), -1).float()
    inter = torch.sum(preds * trues, dim=1)
    dice = torch.mean((2.0 * inter + smooth) / (preds.sum(dim=1) + trues.sum(dim=1) + smooth))
    return dice


def dice_coef_numpy(preds, trues, smooth=1e-3, channel=None):
    if channel is not None:
        preds = preds[:, channel, :, :]
        trues = trues[:, channel, :, :]
    preds = preds.reshape(preds.shape[0], -1)
    trues = trues.reshape(trues.shape[0], -1)

    inter = np.sum(preds * trues, 1)
    dice = np.mean((2.0 * inter + smooth) / (preds.sum(1) + trues.sum(1) + smooth))
    return dice


class SoftDiceCoef(nn.Module):
    def __init__(self, class_id=-1):
        self.class_id = class_id
        super().__init__()

    def forward(self, inputs, target):
        if self.class_id > -1:
            inputs = inputs[:, self.class_id]
            target = target[:, self.class_id]
        inputs = torch.sigmoid(inputs)
        dice = dice_coef(inputs, target)
        return dice


class HardDiceCoef(nn.Module):
    def __init__(self, class_id=None, threshold=0.5):
        super().__init__()
        self.class_id = class_id
        self.threshold = threshold

    def forward(self, inputs, target):
        if self.class_id is not None:
            inputs = inputs[:, self.class_id]
            target = target[:, self.class_id]
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        dice = dice_coef(inputs, target)
        return dice


# Classification
class AccuracyScore(nn.Module):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        acc = accuracy_score(target.cpu().numpy(), np.round(inputs.cpu().numpy()))
        return acc


class JaccardScore(nn.Module):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        acc = jaccard_score(target.cpu().numpy(), np.round(inputs.cpu().numpy()), average='micro')
        return acc
