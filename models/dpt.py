"""
This file contains the classes necessary to implement the DPT fusion model
first published in Ranftl et. al., (Vision Transformers for Dense Prediction)
"""

from omegaconf import DictConfig, ListConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import VisionTransformer


class DPT(nn.Module):
    """
    Baseline model for the DPT depth model
    """

    def __init__(self, config: DictConfig | ListConfig):
        super(DPT, self).__init__()

        # create vision transformer with embedded hooks
        self.vit = VisionTransformer(config)
