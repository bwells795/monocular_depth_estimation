"""
This file contains the classes necessary to implement the DPT fusion model
first published in Ranftl et. al., (Vision Transformers for Dense Prediction)
"""

from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW, Optimizer
from DPT.dpt.models import DPTDepthModel
from losses.mde_losses import ScaleAndShiftInvariantLoss
from losses.LMR import LMRLoss
from torch.utils.data import DataLoader
from dataloaders.nyu_data import NyuDepthV2
from LNRegularizer.LNR import LNR
from torchvision.transforms import v2
from typing import Dict
import sys

if "IPython" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def train_simple(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    epochs: int = 50,
    print_every: int = 1,
) -> None:
    """
    Train depth head on NYU dataset with no additional regularization
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    # loss function
    loss = ScaleAndShiftInvariantLoss()

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
            mse_loss = F.mse_loss(prediction, y)
            l1_loss = F.smooth_l1_loss(prediction, y)

            composite_loss = (2 * err) + (0.5 * mse_loss) + (0.1 * l1_loss)

            # process optimizer
            optim.zero_grad()
            composite_loss.backward()  # back-prop losses
            optim.step()

            # debugging
            grads = []
            for p in model.parameters():
                try:
                    grads.append(p.grad.view(-1))
                except:
                    pass
            grads = torch.cat(grads)

            if i % print_every == 0:
                with torch.no_grad():
                    mse_loss = F.mse_loss(prediction, y)
                pbar.set_postfix_str(
                    f"train_loss: {err:.2f} | mse_loss: {mse_loss:.2f} | l1_loss: {l1_loss:.2f} | composite loss: {composite_loss:.2f} | Average gradient: {grads.mean()} | min grad: {grads.min()} | max grad {grads.max()}"
                )
                pbar.update(print_every)

            i += 1


def train_with_lmr(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    epochs: int = 50,
    print_every: int = 1,
) -> None:
    """
    Train depth head on NYU dataset with lmr regularizer
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    new_size = [x // (2**config.depth.mlp.downsample) for x in config.vit.embed.img_size]
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    lmr_loss = LMRLoss()

    # training loop
    for _ in range(epochs):
        for batch in loader:
            if len(batch["image"])%2 !=0:
                X = batch["image"][:-1].float().to("cuda")
                y = batch["depth"][:-1].float().to("cuda")
                mask = batch["mask"][:-1].float().to("cuda")
            else:
                X = batch["image"].float().to("cuda")
                y = batch["depth"].float().to("cuda")
                mask = batch["mask"].float().to("cuda")

            X = X.permute(0, 3, 1, 2)

            # pass input through LMR model
            output_mask = LNR(X)

            # calculate depth
            prediction = model(X)

            # calculate losses
            err = loss(prediction, y, mask)
            lmr_mask_loss = lmr_loss(
                net_mask=output_mask, depth_hat=prediction.detach(), depth=y, k=100
            )

            err = err + lmr_mask_loss  # combine losses

            # process optimizer
            optim.zero_grad()
            err.backward()  # back-prop losses
            optim.step()

            if i % print_every == 0:
                with torch.no_grad():
                    mse_loss = F.mse_loss(prediction, y)
                pbar.set_postfix_str(
                    f"train_loss: {err:.2f} | mse_loss: {mse_loss:.2f}"
                )
                pbar.update(print_every)

            i += 1


def train_with_cutmix(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    epochs: int = 50,
    print_every: int = 1,
) -> None:
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
                with torch.no_grad():
                    mse_loss = F.mse_loss(prediction, y)
                pbar.set_postfix_str(
                    f"train_loss: {err:.2f} | mse_loss: {mse_loss:.2f}"
                )
                pbar.update(print_every)

            i += 1


def eval(model: nn.Module, loader: DataLoader) -> Dict:
    """
    assess the accuracy of the model
    """
    model.eval()
    pbar = tqdm(total=len(loader), desc="Evaluating MDE:")
    test_err = []
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    new_size = [x // (2**config.depth.mlp.downsample) for x in config.vit.embed.img_size]
    for batch in loader:
        X = batch["image"].float().to("mps")
        y = batch["depth"].float().to("mps")
        with torch.no_grad():
            X = X.permute(0, 3, 1, 2)
            prediction = model(X)
            err = F.mse_loss(
                prediction, y
            )  # when looking at error for evaluation use MSE loss
            test_err.append(err)

        # process optimizer
        pbar.set_postfix_str(f"mse_loss: {err:.2f}")
        pbar.update(1)

    print(f"Average MSE Loss: {sum(test_err)/len(test_err)}")
    return {"mse_avg": sum(test_err) / len(test_err)}


if __name__ == "__main__":
    print("Running training and assessment for DPT-based model")
    config = OmegaConf.load("config/train_config.yaml")

    MDE_model = DPTDepthModel(
        scale=0.000305,
        shift=0.1378,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )  # create a standard DPT depth prediction model# download from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    MDE_model = MDE_model.float().to("mps")
    NYU_DATA_PATH = "data/nyu_data/nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "data/nyu_data/splits.mat"

    nyu_test_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test")
    nyu_train_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train")
    nyu_train_dataloader = DataLoader(nyu_train_ds, batch_size=1)
    nyu_test_dataloader = DataLoader(nyu_train_ds, batch_size=1)

    # create optimizer object
    optim = AdamW(MDE_model.parameters(), lr=0.001, weight_decay=1e-5)

    # train and evaluate model
    train_simple(
        model=MDE_model,
        loader=nyu_train_dataloader,
        optim=optim,
        epochs=1,
    )
    eval(MDE_model, loader=nyu_test_dataloader)
