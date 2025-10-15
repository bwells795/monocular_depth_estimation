"""

Training script for the model
"""

from models.vit import VisionTransformer
from omegaconf import DictConfig, OmegaConf
from models.depth_estimator import DepthEstimator
from diode.diode import DIODE


def main():
    config = OmegaConf.load("config/train_config.yaml")

    mde = DepthEstimator(config=config)
    dataset = DIODE(
        meta_fname="data",
        data_root="data",
        splits="train",
        scene_types="indoors",
    )

    mde.train()
    mde.train_estimator(data_src=dataset)


if __name__ == "__main__":
    main()
