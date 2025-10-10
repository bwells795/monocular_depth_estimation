"""
Model which wraps all of the models plus losses and regularizers with a training loop

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from omegaconf import DictConfig


class DepthEstimator(nn.Module):
    def __init__(self, config: DictConfig):
        # use config to set parameters
        pass

    def forward(self):
        pass

    def train_estimator(self):
        pass

    def infer(self):
        pass
