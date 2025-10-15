"""
Model which wraps all of the models plus losses and regularizers with a training loop

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
import numpy as np

from models.utils import MLP
from models.vit import VisionTransformer
from torch.optim.adam import Adam
from torch.utils.data import Dataset


class DepthEstimator(nn.Module):
    def __init__(self, config: DictConfig | ListConfig):
        super().__init__()
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

        self.train_config = config.train

    def forward(self, imgs: torch.Tensor):
        vit_out = self.vit(imgs)
        depth_map = self.mlp(vit_out)
        # may also need to reshape here
        return depth_map

    def train_estimator(self, data_src: Dataset):
        for e in range(self.train_config.n_epochs):
            for img, gt_depths in data_src:
                # get predicted depth maps
                depths = self(img)

                # convert img format to tensor if needed
                if isinstance(gt_depths, np.ndarray):
                    gt_torch = torch.tensor(gt_depths)
                elif isinstance(gt_depths, torch.Tensor):
                    gt_torch = gt_depths
                else:
                    raise TypeError(
                        f"ground truth data object must be either a numpy ndarray or torch Tensor, got {type(gt_depths)}"
                    )

                loss = F.mse_loss(depths, gt_torch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
