"""
loss file for all the loss functions used in this model
name: loss.py
date: July 2018
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np


class CrossEntropyLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        class_weights = np.load(config.class_weights)
        self.loss = nn.CrossEntropyLoss(ignore_index=config.ignore_index,
                                  weight=torch.from_numpy(class_weights.astype(np.float32)),
                                  size_average=True, reduce=True)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
