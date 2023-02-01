import torch.nn as nn
import torch
import torchsort
from scipy.stats import spearmanr
from torch import Tensor


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
        x1_r = torchsort.soft_rank(x1, regularization_strength=self.regularization_strength, regularization=self.regularization)
        x2_r = torchsort.soft_rank(x2, regularization_strength=self.regularization_strength, regularization=self.regularization)


        x1_c = x1_r - torch.mean(x1_r, dim=self.dim, keepdim=True)
        x2_c = x2_r - torch.mean(x2_r, dim=self.dim, keepdim=True)


        x1_nrm = torch.linalg.norm(x1_c, dim=self.dim)
        x2_nrm = torch.linalg.norm(x2_c, dim=self.dim)

        corr = torch.sum(x1_c * x2_c, dim=-1) / (x1_nrm * x2_nrm)

        loss = 1 - corr

        return loss.reshape(n_sents, trg_length)


if __name__ == "__main__":

    s = 2
    t = 6
    d = 16

    # low reg = exact result; high reg = smoother.
    reg = 0.01

    x1 = torch.randn(s, t, d) #.to(torch.device("cuda"))
    x2 = torch.randn(s, t, d) #.to(torch.device("cuda"))

    print("True spearmanr from scipy:")
    for xx1, xx2 in zip(x1, x2):
        scipy_spearmanr, _ = spearmanr(xx1, xx2)
        print(scipy_spearmanr)
        print(scipy_spearmanr.shape)
    soft_spearman = SoftSpearman(regularization_strength=reg)
    

    print("Soft spearman:")
    soft_spearmanr = soft_spearman(x1, x2)
    print(soft_spearmanr)
    print(soft_spearmanr.shape)




