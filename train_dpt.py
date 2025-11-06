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
from losses.LMR import LMRLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders.nyu_data import NyuDepthV2
from LNRegularizer.LNR import LNR
from torchvision.transforms import v2


def train_simple(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    epochs: int = 50,
    print_every: int = 1,
):
    """
    Train depth head on NYU dataset with no additional regularization
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    lmr_loss = LMRLoss()

    # training loop
    for _ in range(epochs):
        for batch in loader:
            X = batch["image"].float().to("mps")
            y = batch["depth"].float().to("mps")
            mask = batch["mask"].float().to("mps")

            X = X.permute(0, 3, 1, 2)

            # calculate depth
            prediction = model(X)

            # calculate losses
            err = loss(prediction, y, mask)

            # process optimizer
            optim.zero_grad()
            err.backward()  # back-prop losses
            optim.step()

            if i % print_every == 0:
                pbar.set_postfix_str(f"loss: {err:.2f}")
                pbar.update(print_every)

            i += 1


def train_with_lmr(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    epochs: int = 50,
    print_every: int = 1,
):
    """
    Train depth head on NYU dataset with lmr regularizer
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    lmr_loss = LMRLoss()

    # training loop
    for _ in range(epochs):
        for batch in loader:
            X = batch["image"].float().to("mps")
            y = batch["depth"].float().to("mps")
            mask = batch["mask"].float().to("mps")

            X = X.permute(0, 3, 1, 2)

            # pass input through LMR model
            output_mask = LNR(X)

            # calculate depth
            prediction = model(X)

            # calculate losses
            err = loss(prediction, y, mask)
            lmr_mask_loss = lmr_loss(
                net_mask=output_mask, depth_hat=prediction.detatch(), depth=y, k=100
            )

            err = err + lmr_mask_loss  # combine losses

            # process optimizer
            optim.zero_grad()
            err.backward()  # back-prop losses
            optim.step()

            if i % print_every == 0:
                pbar.set_postfix_str(f"loss: {err:.2f}")
                pbar.update(print_every)

            i += 1


def train_with_cutmix(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    epochs: int = 50,
    print_every: int = 1,
):
    """
    Train depth head on NYU dataset using cutmix regularization
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    lmr_loss = LMRLoss()

    # cutmix transform
    cutmix = v2.CutMix()

    # training loop
    for _ in range(epochs):
        for batch in loader:
            X = batch["image"].float().to("mps")
            y = batch["depth"].float().to("mps")
            mask = batch["mask"].float().to("mps")

            X = X.permute(0, 3, 1, 2)

            # apply cutmix to batch
            X, y = cutmix(X, y)

            # calculate depth
            prediction = model(X)

            # calculate losses
            err = loss(prediction, y, mask)
            # process optimizer
            optim.zero_grad()
            err.backward()  # back-prop losses
            optim.step()

            if i % print_every == 0:
                pbar.set_postfix_str(f"loss: {err:.2f}")
                pbar.update(print_every)

            i += 1


def eval(model: nn.Module, loader: DataLoader):
    """
    assess the accuracy of the model
    """
    model.eval()
    pbar = tqdm(total=len(loader), desc="Evaluating MDE:")
    test_err = []
    for batch in loader:
        X = batch["image"]
        y = batch["depth"]
        with torch.no_grad():
            prediction = model(X)
            err = F.mse_loss(
                prediction, y
            )  # when looking at error for evaluation use MSE loss
            test_err.append(err)

        # process optimizer
        pbar.update(1)

    print(f"Average Scale and Shift Inviariant Loss: {sum(test_err)/len(test_err)}")


if __name__ == "__main__":
    print("Running training and assessment for DPT-based model")

    MDE_model = DPTDepthModel()  # create a standard DPT depth prediction model# download from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    MDE_model = MDE_model.float().to("mps")
    NYU_DATA_PATH = "data/nyu_data/nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "data/nyu_data/splits.mat"

    nyu_test_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test")
    nyu_train_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train")
    nyu_train_dataloader = DataLoader(nyu_train_ds, batch_size=12)
    nyu_test_dataloader = DataLoader(nyu_train_ds, batch_size=12)

    # create optimizer object
    optim = Adam(MDE_model.parameters(), lr=0.001)

    # train and evaluate model
    train_simple(
        model=MDE_model,
        loader=nyu_train_dataloader,
        optim=optim,
        epochs=1,
    )
    eval(MDE_model, loader=nyu_test_dataloader)
