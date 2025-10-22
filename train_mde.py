"""

Training script for the model
"""

from torch.utils.data import DataLoader
from models.vit import VisionTransformer
from omegaconf import DictConfig, OmegaConf
from models.depth_estimator import DepthEstimator
from diode.diode import DIODE
import torch


def main():
    config = OmegaConf.load("config/train_config.yaml")
    data_path = ""

    mde = DepthEstimator(config=config, device=torch.device("mps"))
    dataset = DIODE(
        meta_fname="diode/diode_meta.json",
        data_root=data_path,
        splits="train",
        scene_types="indoors",
    )
    DIODE_dataloader = DataLoader(
        dataset, batch_size=config.train.batch_size, shuffle=True
    )
    mde.train()
    mde.train_estimator(data_src=DIODE_dataloader)


if __name__ == "__main__":
    main()
