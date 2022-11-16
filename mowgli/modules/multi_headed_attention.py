import math, torch, torch.nn as nn
from torch import Tensor

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"
    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(
        self,
        num_heads: int,             # the number of heads
        size: int,                  # model size (must be divisible by `num_heads`)
        dropout: float = 0.1,       # probability of dropping a unit
    ):
        super().__init__()
        assert size % num_heads == 0, "`size` must be divisible by `num_heads`"
        self.head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        self.k_layer = nn.Linear(size, num_heads * self.head_size)
        self.v_layer = nn.Linear(size, num_heads * self.head_size)
        self.q_layer = nn.Linear(size, num_heads * self.head_size)
        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self._save_src_trg_context = False # set by MowgliModel
        self.context = None

    def forward(
        self,
        k: Tensor,
        v: Tensor,
        q: Tensor,
        mask: Tensor = None
    ) -> Tensor:
        """
        Computes multi-headed attention.
        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None: scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.head_size)

        if self._save_src_trg_context:
            self.context = context

        return self.output_layer(context)
