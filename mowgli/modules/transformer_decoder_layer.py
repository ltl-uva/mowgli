import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mowgli.modules import MultiHeadedAttention, PositionwiseFeedForward


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.
        It attends to the source representation and the previous decoder states.
        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super().__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size, dropout=dropout)
        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.
        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o
