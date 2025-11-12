"""
This file contains the classes necessary to implement the DPT fusion model
first published in Ranftl et. al., (Vision Transformers for Dense Prediction)
"""

from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import Adam, Optimizer
from DPT.dpt.models import DPTDepthModel
from losses.mde_losses import ScaleAndShiftInvariantLoss
#from losses.LMR import 
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloaders.nyu_data import NyuDepthV2
from models.depth_estimator import DepthEstimator
import torchvision.transforms as v2


def train(
    model: nn.Module,
    loader: DataLoader,
    optim: Optimizer,
    epochs: int = 50,
    print_every: int = 1,
):
    """
    Train depth head on NYU dataset
    """
    model.train()
    pbar = tqdm(total=epochs * len(loader), desc="Training MDE:")
    i = 0
    new_size = [x // (2**config.depth.mlp.downsample) for x in config.vit.embed.img_size]
    # loss function
    loss = ScaleAndShiftInvariantLoss()
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

            prediction = model(X)
            prediction = prediction.reshape(-1, *new_size)
            # to compare depths we need to downsample the ground truth depths by the same downsample factor
            transform = v2.Resize(new_size)
            
            err = loss(prediction, transform(y), transform(mask))

            # process optimizer
            optim.zero_grad()
            err.backward()
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
    # loss function
    loss = ScaleAndShiftInvariantLoss()
    new_size = [x // (2**config.depth.mlp.downsample) for x in config.vit.embed.img_size]
    for batch in loader:
        if len(batch["image"])%2 !=0:
            X = batch["image"][:-1].float().to("cuda")
            y = batch["depth"][:-1].float().to("cuda")
            mask = batch["mask"][:-1].float().to("cuda")
        else:
            X = batch["image"].float().to("cuda")
            y = batch["depth"].float().to("cuda")
            mask = batch["mask"].float().to("cuda")
            
        
        with torch.no_grad():
            X = X.permute(0, 3, 1, 2)
            prediction = model(X)
            prediction = prediction.reshape(-1, *new_size)
            # to compare depths we need to downsample the ground truth depths by the same downsample factor
            transform = v2.Resize(new_size)
            err = loss(prediction, transform(y), transform(mask))
            test_err.append(err)

        # process optimizer
        pbar.update(1)

    print(f"Average Scale and Shift Inviariant Loss: {sum(test_err)/len(test_err)}")


if __name__ == "__main__":
    print("Running training and assessment for DPT-based model")
    config = OmegaConf.load("config/train_config.yaml")

    MDE_model = DepthEstimator(config=config, device=torch.device("cuda"), use_LNR=True)
    # download from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    NYU_DATA_PATH = "data/nyu_data/nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "data/nyu_data/splits.mat"

    nyu_test_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test")
    nyu_train_ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="train")
    nyu_train_dataloader = DataLoader(nyu_train_ds, batch_size=config.train.batch_size, shuffle=True)
    nyu_test_dataloader = DataLoader(nyu_train_ds, batch_size=config.train.batch_size, shuffle=True)

    # create optimizer object
    optim = Adam(MDE_model.parameters(), lr=0.001)

    # train and evaluate model
    train(
        model=MDE_model,
        loader=nyu_train_dataloader,
        optim=optim,
        epochs=1,
    )
    eval(MDE_model, loader=nyu_test_dataloader)
