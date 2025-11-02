from omegaconf import DictConfig, ListConfig
from models.utils import Embedding, Encoding, TransformerEncoder
import torch.nn as nn
import torch


class VisionTransformer(nn.Module):
    """
    Implementation of the vision transformer which can be used as the backbone of various vision models
    - This model does not include the final MLP which actually 'learns' the current task
    """

    def __init__(self, config: DictConfig | ListConfig):
        super().__init__()

        self.embed = Embedding(
            img_size=config.vit.embed.img_size,
            patch_size=config.vit.embed.patch_size,
            in_channels=config.vit.embed.in_channels,
            out_channels=config.vit.model_dim,
        )

        # calculate sequence length from patching
        n_patches = (
            config.vit.embed.img_size[0] * config.vit.embed.img_size[1]
        ) / config.vit.embed.patch_size**2

        self.encoder = Encoding(
            out_channels=config.vit.model_dim,
            sequence_length=int(n_patches) + 1,
        )

        self.tfm_backbone = nn.Sequential(
            *[
                TransformerEncoder(config)
                for _ in range(config.vit.transformer.tfm_depth)
            ]
        )

        # for i, layer in enumerate(self.tfm_backbone):
        #     # register hooks at hook locations from config
        #     if i in config.hook_layers:
        #         layer.register_forward_hook()

    def forward(self, img: torch.Tensor):
        """
        process an image through the vision transformer
        """
        img_proc = self.encoder(self.embed(img))
        tfm_out = self.tfm_backbone(img_proc)
        return tfm_out  # return the output from the transformer layer
