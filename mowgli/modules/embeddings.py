import math
from torch import nn, Tensor
from mowgli.helpers import freeze_params


class Embeddings(nn.Module):

    """
    Simple embeddings class
    """
    def __init__(self, embedding_dim: int, scale: bool, vocab_size: int, padding_idx: int, freeze: bool = False, **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)

        return self.lut(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, vocab_size={self.vocab_size})"
