import torch
import torch.nn as nn


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt)**self.gamma * torch.log(pt)).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        batch_size = input.size(0)
        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1)

        # compute the actual dice score
        intersection = torch.sum(input * target, dim=1)
        fps = torch.sum(input * (1 - target), dim=1)
        fns = torch.sum((1 - input) * target, dim=1)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_score = (numerator + self.eps) / (denominator + self.eps)
        tversky_loss = (1 - tversky_score) ** self.gamma
        return tversky_loss.mean()