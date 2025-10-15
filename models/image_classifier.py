from typing_extensions import List
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


class Classifier(nn.Module):
    def __init__(self, config: DictConfig | ListConfig, device: torch.device):
        super().__init__()
        # use config to set parameters
        self.vit = VisionTransformer(config)
        self.mlp = MLP(
            n_hidden_layers=config.head.mlp.n_hidden_layers,
            n_inputs=config.head.mlp.n_inputs,
            n_outputs=config.head.mlp.n_outputs,
            n_hidden_nodes=config.head.mlp.n_hidden_nodes,
            activation=config.head.mlp.activation,
        )
        self.optimizer = Adam(self.parameters(), lr=config.train.lr)

        self.train_config = config.train

        self.loss = nn.CrossEntropyLoss()
        self.device = device

        # move the model to the specified device
        self.float().to(device)

    def forward(self, imgs: torch.Tensor):
        vit_out = self.vit(imgs)

        # use the encoding layer only
        c = self.mlp(vit_out[:, 0])
        return c

    def train_model(self, data_src: DataLoader):
        # set the model to train mode
        self.train()

        # do the training loop
        pbar = tqdm(total=len(data_src) * self.train_config.n_epochs)
        for e in range(self.train_config.n_epochs):
            for img, label in data_src:
                img = img.to(self.device)
                label = label.to(self.device)
                # get predicted depth maps
                y_hat = self(img)
                loss = self.loss(y_hat, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix_str(f"epoch: {e} | loss: {loss:.4f}")
                pbar.update(1)

    def evaluate(self, test: DataLoader):
        self.eval()

        correct, total = 0, 0
        for img, label in test:
            img = img.to(self.device)
            label = label.to(self.device)
            # get predicted depth maps
            y_hat = self(img)

            correct += sum(torch.argmax(F.softmax(y_hat, dim=-1), dim=1) == label)
            total += len(img)

        return correct / total
