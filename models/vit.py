from omegaconf import DictConfig
from utils import Embedding, Encoding, TransformerEncoder
import torch.nn as nn
import torch


class VisionTransformer(nn.Module):
    """
    Implementation of the vision transformer which can be used as the backbone of various vision models
    - This model does not include the final MLP which actually 'learns' the current task
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        self.embed = Embedding(
            img_size=config.embed.img_size,
            patch_size=config.embed.patch_size,
            in_channels=config.embed.in_channels,
            out_channels=config.embed.out_channels,
        )
        self.encoder = Encoding(
            out_channels=config.encode.model_dim, sequence_length=config.encode.seq_len
        )

        self.tfe = TransformerEncoder(config)
        self.tfm_backbone = nn.Sequential(
            nn.ModuleList([self.tfe for _ in config.vit.transformer.tfm_depth])
        )

    def forward(self, img: torch.Tensor):
        """
        process an image through the vision transformer
        """
        img_proc = self.encoder(self.embed(img))
        tfm_out = self.tfm_backbone(img_proc)
        return tfm_out  # return the output from the transformer layer
