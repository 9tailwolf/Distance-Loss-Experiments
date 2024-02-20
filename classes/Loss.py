import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
from distanceloss import DiMSLoss, ADiMSLoss

from classes.Utils import get_device
    
class DiMALoss(nn.Module):
    def __init__(self):
        super(DiMALoss, self).__init__()
        self.device = get_device()

    def compute_weights(self, target, indices):
        target_index = torch.argmax(target,dim=1).view(-1,1)
        weights = 1+torch.abs(indices - target_index)
        weights = weights ** 2
        return weights

    def forward(self, predicted, target):
        indices = torch.arange(target.size(1)).unsqueeze(0).repeat(target.size(0),1).to(self.device)
        weights = self.compute_weights(target, indices)
        mse_loss = torch.mean(weights * abs(predicted - target), dim=1)
        mse_loss = mse_loss.sum()

        return mse_loss
    
class ADiMALoss(nn.Module):
    def __init__(self):
        super(ADiMALoss, self).__init__()
        self.device = get_device()

    def compute_weights(self, target, indices):
        target_index = torch.argmax(target,dim=1).view(-1,1)
        weights = 1+torch.abs(indices - target_index)
        return weights

    def forward(self, predicted, target):
        indices = torch.arange(target.size(1)).unsqueeze(0).repeat(target.size(0),1).to(self.device)
        weights = self.compute_weights(target, indices)
        mse_loss = torch.mean(weights * abs(predicted - target), dim=1)
        mse_loss = mse_loss.sum()

        return mse_loss