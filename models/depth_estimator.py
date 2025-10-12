"""
Model which wraps all of the models plus losses and regularizers with a training loop

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from omegaconf import DictConfig
import numpy as np

from models.utils import MLP
from models.vit import VisionTransformer
from torch.optim.adam import Adam


class DepthEstimator(nn.Module):
    def __init__(self, config: DictConfig):
        # use config to set parameters
        self.vit = VisionTransformer(config)
        self.mlp = MLP(
            n_hidden_layers=config.depth.mlp.n_hidden_layers,
            n_inputs=config.depth.mlp.n_inputs,
            n_outputs=config.depth.mlp.n_outputs,
            n_hidden_nodes=config.depth.mlp.n_hidden_nodes,
            activation=config.depth.mlp.activation,
        )
        self.optimizer = Adam(self.parameters())

    def forward(self, imgs: torch.Tensor):
        vit_out = self.vit(imgs)
        depth_map = self.mlp(vit_out)
        return depth_map

    def train_estimator(self, imgs: torch.Tensor, gt_depths: np.ndarray | torch.Tensor):
        # get predicted depth maps
        depths = self(imgs)

        # convert img format to tensor if needed
        if isinstance(gt_depths, np.ndarray):
            gt_torch = torch.tensor(gt_depths)
        elif isinstance(gt_depths, torch.Tensor):
            gt_torch = gt_depths

        loss = F.mse_loss(depths, gt_torch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def infer(self):
        pass
