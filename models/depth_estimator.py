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
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as v2


class DepthEstimator(nn.Module):
    def __init__(self, config: DictConfig | ListConfig, device: torch.device):
        super().__init__()
        # use config to set parameters
        self.vit = VisionTransformer(config)

        # determine the ouput size based off of input image size and downsample
        self.ds = config.depth.mlp.downsample
        img_size = config.vit.embed.img_size
        self.new_size = [x // (2**self.ds) for x in img_size]
        n_outputs = torch.prod(torch.tensor(self.new_size), dtype=torch.float32)
        n_outputs = int(n_outputs.item())

        self.mlp = MLP(
            n_hidden_layers=config.depth.mlp.n_hidden_layers,
            n_inputs=config.depth.mlp.n_inputs,
            n_outputs=n_outputs,
            n_hidden_nodes=config.depth.mlp.n_hidden_nodes,
            activation=config.depth.mlp.activation,
        )
        self.optimizer = Adam(self.parameters())

        self.train_config = config.train
        self.device = device
        self.to(self.device)

    def forward(self, imgs: torch.Tensor):
        vit_out = self.vit(imgs)
        depth_map = self.mlp(vit_out[:, 0])
        # may also need to reshape here
        return depth_map

    def train_estimator(self, data_src: DataLoader):
        self.train()

        # create training loop and make progress bar
        pbar = tqdm(total=len(data_src) * self.train_config.n_epochs)
        for e in range(self.train_config.n_epochs):
            for img, gt_depths, _ in data_src:
                # convert img format to tensor if needed
                if isinstance(gt_depths, np.ndarray):
                    gt_torch = torch.tensor(gt_depths)
                elif isinstance(gt_depths, torch.Tensor):
                    gt_torch = gt_depths
                else:
                    raise TypeError(
                        f"ground truth data object must be either a numpy ndarray or torch Tensor, got {type(gt_depths)}"
                    )

                # move model to device
                img = img.to(self.device).float()
                gt_torch = gt_torch.to(self.device).float()

                # permute image and depth data to match the expected shape for the transformer [B, C, H, W]
                img = img.permute(0, 3, 1, 2)

                # get predicted depth maps
                depths = self(img)
                depths = depths.reshape(-1, *self.new_size)

                # to compare depths we need to downsample the ground truth depths by the same downsample factor
                transform = v2.Resize(self.new_size)
                gt = transform(gt_torch)

                loss = F.mse_loss(depths, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # provide some clarity into memory usage during training
                curr_mem = 0
                if self.device.type == "mps":
                    curr_mem = torch.mps.current_allocated_memory()
                elif self.device == "cuda":
                    _, curr_mem = torch.cuda.mem_get_info(self.device)

                pbar.set_postfix_str(
                    f"epoch {e} | loss: {loss:.4f} | Memory in use: {(curr_mem/1e9):.2f} GB"
                )
                pbar.update(1)

    def evaluate(self, test_data: DataLoader):
        """
        Method which calculates model statistics over a given test/validation dataset
        """
        for img, gt_data in test_data:
            img.to(self.device)
            d_hat = self(img)
