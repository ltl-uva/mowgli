import torch
from torch import nn, Tensor
from torch.autograd import Variable

class LabelSmoothedCrossEntropyLoss(nn.Module):
    """Cross-Entropy Loss with optional label smoothing."""
    def __init__(self, pad_idx: int, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        if self.smoothing <= 0.0:
            self.criterion = nn.NLLLoss(ignore_index=self.pad_idx, reduction="sum") # standard xent loss
        else:
            self.criterion = nn.KLDivLoss(reduction="sum") # label-smoothed loss using KL divergence loss

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """Smooths target distribution. All non-reference words get uniform probability mass according to `self.smoothing`."""
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float() # batch*seq_len x vocab_size
        smooth_dist.fill_(self.smoothing / (vocab_size - 2)) # fill distribution uniformly with smoothing
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)# assign true label the probability of 1-smoothing
        smooth_dist[:, self.pad_idx] = 0 # give padding probability of 0 everywhere
        padding_positions = torch.nonzero(targets.data == self.pad_idx, as_tuple=False) # masking out padding area
        if len(padding_positions) > 0: smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0: targets = self._smooth_targets(targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1))
        else:                  targets = targets.contiguous().view(-1)
        assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape == targets.shape
        return self.criterion(log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
