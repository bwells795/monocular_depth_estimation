import torch
import torch.nn as nn
from typing import Tuple
from omegaconf import DictConfig


class Embedding(nn.Module):
    """
    Class which contains the convolutional layer to convert a full image into downsampled patches for encoding as a 2D vector
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_size: int,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        assert img_size[0] % patch_size == 0
        assert img_size[1] % patch_size == 0

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # use a convolutional layer to extract pixel patches from the provided image without overlap
        # embed the patched kernels to the model dimension out_channels
        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor):
        x = self.downsample(x)
        x.flatten(2)
        x.transpose(1, 2)
        return x


class Encoding(nn.Module):
    """
    Class containing the emcoding representation
    """

    pos_encs: torch.Tensor

    def __init__(self, out_channels, sequence_length):
        super().__init__()

        # create class labels for each out channel
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, out_channels))

        # create position encodings which map each input to a unique position identifier
        pos_encs = torch.zeros(sequence_length, out_channels)
        for i in range(sequence_length):
            for j in range(out_channels):
                if j % 2 == 0:
                    pos_encs[i][j] = torch.sin(i / 10000 ** (2 * j / out_channels))
                else:
                    pos_encs[i][j] = torch.cos(
                        i / 10000 ** (2 * (j - 1) / out_channels)
                    )

        self.register_buffer("pos_encs", pos_encs.unsqueeze(0))

    def forward(self, x):
        # generate class tokens for each image in the batch
        tokens = self.class_token.expand(x.shape[0], -1, -1)

        # add the class labels to the beginning of each batch
        x = torch.cat([tokens, x], dim=1)

        # add positional embeddings to the tokens
        x = x + self.pos_encs

        return x


class SelfAttention(nn.Module):
    """
    Representation of the Scaled Dot-Product Attention module from the paper
    "Attention Is All You Need"

    """

    def __init__(self, n_head_nodes: int, num_channels: int):
        super().__init__()
        self.n_head_nodes = n_head_nodes
        self.num_channels = num_channels

        self.Q = nn.Linear(n_head_nodes, num_channels)
        self.K = nn.Linear(n_head_nodes, num_channels)
        self.V = nn.Linear(n_head_nodes, num_channels)

    def forward(self, x):
        """
        Implement torch functional dot product attention
        """
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        attn_res = torch.nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v
        )

        return attn_res


class MultiheadAttention(nn.Module):
    """
    Representation of multi-head attention from Attention is All You Need

    """

    def __init__(self, n_total_nodes, n_heads, n_channels):
        super().__init__()
        assert (
            n_total_nodes % n_heads == 0
        ), f"Number of multi-head attention heads must be divisible by the number of heads: received {n_total_nodes} total nodes and {n_heads}"

        self.nodes_per_head = n_total_nodes / n_heads

        self.heads = nn.ModuleList(
            [
                SelfAttention(n_head_nodes=self.nodes_per_head, num_channels=n_channels)
                for _ in range(n_heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        all_procs = []
        for head in self.heads:
            all_procs.append(head(x))
        return torch.cat(all_procs)


class MLP(nn.Module):
    """
    Simple fully connected MLP with dropout and configurable activation
    """

    def __init__(
        self, n_hidden_layers, n_inputs, n_outputs, n_hidden_nodes, activation: str
    ):
        super().__init__()

        assert n_hidden_layers >= 1, f"MLP does not support less than one hidden layer"
        self.first = nn.Linear(n_inputs, n_hidden_nodes)
        self.middle = nn.Linear(n_hidden_nodes, n_hidden_nodes)
        self.last = nn.Linear(n_hidden_nodes, n_outputs)

        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "GELU":
            self.activation = nn.GELU()

        self.dropout = nn.Dropout(0.2)

        ## build model from parts
        modules = []
        modules.append(self.first)
        modules.append(self.relu)
        for _ in range(n_hidden_layers):
            modules.append(self.middle)
            modules.append(self.activation)
            modules.append(self.dropout)
        modules.append(self.last)

        mod_list = nn.ModuleList(modules)
        self.model = nn.Sequential(mod_list)

    def forward(self, x):
        """
        use ModuleList directly to pass input through all layers of the MLP
        """
        return self.model(x)


class TransformerEncoder(nn.Module):
    """
    Implementation of the full vision transformer inspired by the encoder transformer from Attention is all you need
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        self.norm = nn.LayerNorm(config.model.transformer.dim)
        self.mlp = MLP(
            n_hidden_layers=config.model.mlp.n_hidden_layers,
            n_inputs=config.model.mlp.n_inputs,
            n_outputs=config.model.mlp.n_outputs,
            n_hidden_nodes=config.model.mlp.n_hidden_nodes,
            activation=config.model.mlp.activation,
        )
        self.mha = MultiheadAttention(
            n_total_nodes=config.vit.transformer.dim,
            n_heads=config.vit.transformer.n_heads,
            n_channels=config.vit.transformer.n_channels,
        )

    def forward(self, x: torch.Tensor):
        """
        Process encoded images keeping skip-level connections from the paper
            - here x represents the encoded image from the vision encoder forward method

        """
        n1 = self.norm(x)
        attn = self.mha(n1)

        # intermediate results also include skip connection from the encoder
        middle = x + attn

        n2 = self.norm(middle)
        mlp_out = self.mlp(n2)

        final = middle + mlp_out

        return final
