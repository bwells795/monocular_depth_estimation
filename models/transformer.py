from utils import MLP, Embedding, MultiheadAttention, Encoding
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as v2
import numpy as np
from typing import Tuple


class VisionTransformer(nn.Module):
    """
    Implementation of the full vision transformer inspired by the encoder transformer from Attention is all you need
    """

    def __init__(self):
        super().__init__()
