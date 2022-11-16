import math

import torch
import torch.distributed as dist

from mowgli.data import MowgliDataset


class DistributedDataset(MowgliDataset):
    def __init__(
        self,
        dataset,
        world_size=None,
        rank=None,
        shuffle=True,
        drop_last=False
    ):
        """ Divides a dataset into `world_size` chunks.
        """
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires torch.distributed package to be available."
                )
            world_size = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires torch.distributed package to be available."
                )
            rank = dist.get_rank()

        self.data = dataset.data
        self.src = dataset.src
        self.trg = dataset.trg

        # if the data length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if drop_last and len(self.data) % world_size != 0:
            # split to nearest available length that is evenly divisible.
            num_samples = math.ceil(
                (len(self.data) - world_size) / world_size
            )
        else:
            num_samples = math.ceil(len(self.data) / world_size)

        total_size = num_samples * world_size

        if shuffle:
            g = torch.Generator()
            indices = torch.randperm(len(self.data), generator=g).tolist()
        else:
            indices = list(range(len(self.data)))

        if not drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible
            indices = indices[:total_size]
        assert len(indices) == total_size

        # subsample
        indices = indices[rank:total_size:world_size]
        assert len(indices) == num_samples

        # only store relevant indices
        self.data = [self.data[i] for i in indices]
        self.src = [self.src[i] for i in indices]
        self.trg = [self.trg[i] for i in indices]
