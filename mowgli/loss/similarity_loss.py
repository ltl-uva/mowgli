import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from fast_soft_sort.pytorch_ops import soft_rank


class SimilarityLoss(nn.Module):
    """Distance loss, used to explicitly align multilingual representations."""
    def __init__(self, pad_idx: int, sim="euclidean", norm=None, layers=[0,1,2,3,4,5], l2=0.01):
        super().__init__()
        self.pad_idx = pad_idx

        assert norm in [None, "mean", "mean_per_t"]

        if   sim == "cosine":               self.sim_fn = CosineDistance(l2=l2, norm=norm)
        elif sim == "euclidean":            self.sim_fn = EuclideanSimilarity()
        elif sim == "layernorm_euclidean":  self.sim_fn = LayerNormEuclideanSimilarity()
        elif sim == "normalized_euclidean": self.sim_fn = NormalizedEuclideanSimilarity()
        elif sim == "soft_spearman":        self.sim_fn = SoftSpearman()
        else:                               raise NotImplementedError

        self.layers = layers


    def forward(self, x1, x2, lang1, lang2, mask=None) -> Tensor:
        """Computes cosine similarity loss between two encoder-decoder attention Tensors."""
        assert all([x1_i.shape == x2_i.shape for x1_i, x2_i in zip(x1, x2)])

        sim_loss = {}
        # Get loss for all decoder layers
        for layer_i, (x1_i, x2_i) in enumerate(zip(x1, x2)):
            if layer_i not in self.layers:
                continue

            sim_loss_i = self.sim_fn(x1=x1_i, x2=x2_i, mask=mask)

            # ignore padding
            if mask is not None:
                sim_loss_i = sim_loss_i.masked_fill(mask, 0.0)

            # permute batch and apply negative loss, pushing apart representations for different target tokens
            # apply_contrastive = False
            # if apply_contrastive:
                # shuffled_x1_i, shuffled_mask1 = self._shuffle(x1_i, mask=mask)
                # shuffled_x2_i, shuffled_mask2 = self._shuffle(x2_i, mask=mask)
                
                # shuffled_loss_i = self.sim_fn(shuffled_x1_i, shuffled_x2_i, minimize=False)
                
                # ignore padding
                # if mask is not None:
                    # shuffled_loss_i = shuffled_loss_i.masked_fill(shuffled_mask1 + shuffled_mask2, 0.0)

                # sim_loss_i += shuffled_loss_i

            sim_loss_i = sim_loss_i.sum()
            sim_loss[layer_i] = sim_loss_i

        return sim_loss

    # def _shuffle(self, x: Tensor, mask: Tensor) -> Tensor:
    #     """
    #     Shuffles a tensor along dimension 1 (tokens).
    #     Note that unfortunately there is no efficient torch batch operation to do this, which is why we use a loop.
    #     """

    #     # create different token permutations per sentence, keeping hidden dim order constant
    #     indices = torch.stack([torch.randperm(x.size(1), device=x.device) for _ in range(x.size(0))])
        
    #     # shuffle x and mask
    #     shuffled_x    = torch.stack([x_idx[indices[idx]] for idx, x_idx in enumerate(x)])
        
    #     if mask is not None:
    #         shuffled_mask = torch.stack([m_idx[indices[idx]] for idx, m_idx in enumerate(mask)])
    #     else:
    #         shuffled_mask = None

    #     return shuffled_x, shuffled_mask


class CosineDistance(nn.Module):
    def __init__(self, l2, norm=None, dim=-1, dropout=0.5):
        super().__init__()
        self.l2 = l2
        self.dim = dim
        self.dropout = dropout
        self.cos_sim = nn.CosineSimilarity(dim=self.dim)

        self.subtract_mean   = True if norm == "mean"       else False
        self.subtract_t_mean = True if norm == "mean_per_t" else False

        assert (
            (not self.subtract_mean and not self.subtract_t_mean)
            or
            (self.subtract_mean ^ self.subtract_t_mean)
        )



    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor, minimize=True) -> Tensor:
        """
        Calculates cosine distance with dropout.
        """
        n_sents, trg_length = x1.shape[0], x1.shape[1]

        print(self.subtract_mean, self.subtract_t_mean)

        # reshape
        x1 = x1.contiguous().view(n_sents*trg_length, self.dim)
        x2 = x2.contiguous().view(n_sents*trg_length, self.dim)

        if mask is not None:
            mask = mask.contiguous().view(n_sents*trg_length, 1)
            # set hidden_dim to all 0 if corresponds to masked token
            x1 = torch.where(mask, x1*0, x1)
            x2 = torch.where(mask, x2*0, x2)

            # subtract total mean,
            # i.e. subtract same `hidden_dim`-shaped mean vector for all tokens.
            if self.subtract_mean:
                # divide total sum only by #non-masked tokens
                denominator = sum(mask.eq(False))
                x1 -= torch.sum(x1, dim=0) / denominator
                x2 -= torch.sum(x2, dim=0) / denominator

            # subtract unique mean per dimension
            elif self.subtract_t_mean:
                # reshape back, since we take a column  view
                x1 = x1.contiguous().view(n_sents, trg_length, self.dim)
                x2 = x2.contiguous().view(n_sents, trg_length, self.dim)
                mask = mask.contiguous().view(n_sents, trg_length, 1)

                # divide column sum only by #non-masked tokens in column
                denominator = (n_sents - torch.sum(mask, dim=0))
                x1 -= torch.sum(x1, dim=0) / denominator
                x2 -= torch.sum(x2, dim=0) / denominator

                # reshape again
                x1 = x1.contiguous().view(n_sents*trg_length, self.dim)
                x2 = x2.contiguous().view(n_sents*trg_length, self.dim)


        # apply same dropout mask to x1 and x2
        # dropout_mask = self._get_dropout_mask(dim1=x.shape[0], dim2=x.shape[1], device=x.device)
        # x1_masked, x2_masked = x1*dropout_mask, x2*dropout_mask

        # cosine distance
        loss = self.cos_sim(x1, x2).reshape(n_sents, trg_length)

        # add l2 loss
        # loss += self.l2 * (x1_masked.pow(2.0).sum(dim=-1) + y_masked.pow(2.0).sum(dim=-1))

        return 1 - loss if minimize else 1 + loss


    def _get_dropout_mask(self, dim1: int, dim2: int, device: torch.device) -> Tensor:
        a = torch.ones(dim1, dim2, device=device) * self.dropout
        return torch.bernoulli(a)


class EuclideanSimilarity(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.pwdist = nn.PairwiseDistance(p=p)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        "Computes Euclidean Distance between Tensors `x` and `y`."
        return torch.stack([self.pwdist(x_i, y_i) for x_i, y_i in zip(x,y)])


class LayerNormEuclideanSimilarity(nn.Module):
    def __init__(self, dim: int = 512, p: int = 2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.pwdist = nn.PairwiseDistance(p=p)

    def forward(self, x, y):
        "Computes Euclidean Distance between Tensors `x` and `y`, normalized by LayerNorm to decrease the role of vector magnitude."
        return torch.stack([self.pwdist(self.layer_norm(x_i), self.layer_norm(y_i)) for x_i, y_i in zip(x,y)])


class NormalizedEuclideanSimilarity(nn.Module):
    def __init__(self, dim=2, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        ned_2 = 0.5 * ((x1 - x2).var(dim=self.dim) / (x1.var(dim=self.dim) + x2.var(dim=self.dim) + self.eps))
        return ned_2 ** 0.5


class SoftSpearman(nn.Module):
    def __init__(self, dim: int = -1, regularization_strength: int = 0.1, regularization: str = "kl"):
        super().__init__()
        self.dim = dim
        self.regularization_strength = regularization_strength
        self.regularization = regularization


    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Calculates soft spearman."""
        n_sents, trg_length = x1.shape[0], x1.shape[1]
        
        # n_sents x trg_length x hidden_dim -> n_sents*trg_length x hidden_dim
        x1 = x1.contiguous().view(-1, x1.size(self.dim))
        x2 = x2.contiguous().view(-1, x2.size(self.dim))

        # manually compute spearman
        x1_r = self._soft_rank(x1)
        x2_r = self._soft_rank(x2)

        x1_c = x1_r - torch.mean(x1_r, dim=self.dim, keepdim=True)
        x2_c = x2_r - torch.mean(x2_r, dim=self.dim, keepdim=True)


        x1_nrm = torch.linalg.norm(x1_c, dim=self.dim)
        x2_nrm = torch.linalg.norm(x2_c, dim=self.dim)

        corr = torch.sum(x1_c * x2_c, dim=-1) / (x1_nrm * x2_nrm)

        loss = 1 - corr

        return loss.reshape(n_sents, trg_length)

    def _soft_rank(self, x: Tensor) -> Tensor:
        """Wrapper since soft rank package does not support GPU"""
        device = x.device
        return soft_rank(
            x.cpu(),
            regularization_strength=self.regularization_strength,
            regularization=self.regularization
        ).to(device)
