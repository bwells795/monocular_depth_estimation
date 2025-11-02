"""
This file contains the classes necessary to implement the DPT fusion model
first published in Ranftl et. al., (Vision Transformers for Dense Prediction)
"""

from omegaconf import DictConfig, ListConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import Adam, Optimizer
from DPT.dpt.models import DPTDepthModel
from losses.mde_losses import ScaleAndShiftInvariantLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders.nyu_data import NyuDepthV2


def train(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    loss: nn.Module,
    epochs: int = 50,
):
    """
    Train depth head on NYU dataset
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Trainint MDE:")
    i = 0
    for _ in range(epochs):
        for batch in loader:
            X = batch["image"].to("mps")
            y = batch["depth"].to("mps")
            mask = batch["mask"].to("mps")

            X = X.permute(0, 3, 1, 2)

            prediction = model(X)
            err = loss(prediction, y, mask)

            # process optimizer
            optim.zero_grad()
            err.backward()
            optim.step()

            if i % 10 == 0:
                pbar.set_postfix_str(f"loss: {err:.2f}")
                pbar.update(10)

            i += 1


def eval(model: nn.Module, loader: DataLoader, loss: nn.Module):
    """
    assess the accuracy of the model
    """
    model.eval()
    pbar = tqdm(total=len(loader), desc="Evaluating MDE:")
    for batch in loader:
        X = batch["image"]
        y = batch["depth"]
        mask = batch["mask"]

        prediction = model(X)
        err = loss(prediction, y, mask)

        # process optimizer
        optim.zero_grad()
        err.backward()
        optim.step()
        pbar.update(1)


if __name__ == "__main__":
    print("Running training and assessment for DPT-based model")

    MDE_model = DPTDepthModel()  # create a standard DPT depth prediction model# download from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    NYU_DATA_PATH = "data/nyu_data/nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "data/nyu_data/splits.mat"

    nyu_test_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test")
    nyu_train_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train")
    nyu_train_dataloader = DataLoader(nyu_train_ds, batch_size=32)
    nyu_test_dataloader = DataLoader(nyu_train_ds, batch_size=32)

    # create optimizer object
    optim = Adam(MDE_model.parameters(), lr=0.01)

    # loss function
    credential = ScaleAndShiftInvariantLoss()

    # train and evaluate model
    train(MDE_model, loader=nyu_train_dataloader, optim=optim, loss=credential)
    eval(MDE_model, loader=nyu_test_dataloader, loss=credential)
