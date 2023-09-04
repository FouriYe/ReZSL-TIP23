import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal as MVN

class _Loss(nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class weighted_RegressLoss(_Loss):
    def __init__(self, RegNorm, RegType, device):
        super(weighted_RegressLoss, self).__init__()
        self.RegNorm = RegNorm
        self.RegType = RegType
        self.device = device
        init_noise_sigma = 0.1
        if self.RegType == "BMC":
            self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma)).to(self.device)
    def forward(self, inputs, targets, weights=None, reduction='mean'):
        """
        Args:
            inputs: [n,d]
            targets: [n,d]
            weights: [n,d]
        Return:
            loss: A float tensor.
        """

        if self.RegNorm == True:
            inputs_norm = inputs.norm(p=2, dim=1, keepdim=True).expand_as(inputs)
            inputs = inputs / (inputs_norm + 1e-10)
            targets_norm = targets.norm(p=2, dim=1, keepdim=True).expand_as(targets)
            targets = targets / (targets_norm + 1e-10)
        if self.RegType == "BMC":
            noise_var = self.noise_sigma ** 2
            I = torch.eye(inputs.shape[-1]).to(self.device) # [312,312]
            logits = MVN(inputs.unsqueeze(1), noise_var * I).log_prob(targets.unsqueeze(0)) # logits [batch, batch]
            loss = F.cross_entropy(logits, torch.arange(inputs.shape[0]).to(self.device))
            loss = loss * (2 * noise_var)#.detach()
        elif self.RegType == "MSE":
            offset = (inputs - targets) ** 2
            if weights is not None:
                offset = torch.mul(offset, weights)
            loss = self._get_(offset,reduction)
        elif self.RegType == "RMSE":
            offset = (inputs - targets) ** 2
            offset = torch.sqrt(offset+1e-10)
            if weights is not None:
                offset = torch.mul(offset, weights)
            loss = self._get_(offset,reduction)
        elif self.RegType == "MAE":
            offset = torch.abs(inputs - targets)
            if weights is not None:
                offset = torch.mul(offset, weights)
            loss = self._get_(offset,reduction)
        else:
            raise TypeError(self.RegType+"is not implemented")


        return loss

    def _get_(self, input, reduction):
        if reduction == 'mean':
            output = torch.mean(input)
        elif reduction == 'sum':
            output = torch.sum(input)
        else:
            output = input

        return output
