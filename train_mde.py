"""

Training script for the model
"""

from models.transformer import VisionTransformer
from omegaconf import DictConfig, OmegaConf
from models.depth_estimator import DepthEstimator


def main():
    config = OmegaConf.load("config/train_config.yaml")

    mde = DepthEstimator(config)

    mde.train()
    mde.train_estimator()  # placeholder for now
