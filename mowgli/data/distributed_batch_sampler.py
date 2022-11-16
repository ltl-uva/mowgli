import numpy as np
from typing import Optional, Tuple

import torch

from mowgli.data import LogosBatchSampler


class DistributedBatchSampler:
    """
    Defines a wrapper around a batch sampler that handles distributing a batch
    to multiple workers.
    """

    def __init__(
        self,
        batch_sampler: LogosBatchSampler,
        world_size: Optional[int] = None,
        rank: Optional[int] = None
    ):
        """
        Initializes a distributed batch sampler.

        :param batch_sampler: to be distributed batch sampler
        :param world_size: number of parallel processes, typically the number
            of gpus
        :param rank: rank of active process
        """
        self.batch_sampler = batch_sampler
        self.world_size = world_size
        self.rank = rank

        if self.world_size is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires torch.distributed package to be available."
                )
            self.world_size = torch.distributed.get_world_size()

        if self.rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires torch.distributed package to be available."
                )
            self.rank = torch.distributed.get_rank()

        assert self.rank <= self.world_size

    def __iter__(self) -> None:
        """
        Iterates over batches and returns the distributed {start, end} splits.
        """
        for batch in self.batch_sampler:
            # Ignore batch if it has insufficient instances to distribute
            # this can (only) happen at the very end of the epoch
            if len(batch) >= self.world_size:
                start, end = self.distributed_batch_split(batch)
                yield batch[start:end]

    def distributed_batch_split(self, batch: list) -> Tuple[int, int]:
        """
        Calculates the {start, end} splits for a batch given the active rank.

        :param batch: to be distributed batch, which is a list of integers
            corresponding to DataSet indices.
        :return: batch {start, end} splits for active rank
        """
        chunk_size, remainder = divmod(len(batch), self.world_size)
        section_sizes = (
            [0] + remainder * [chunk_size+1] +
            (self.world_size-remainder) * [chunk_size]
        )
        div_points = np.array(section_sizes).cumsum()

        return div_points[self.rank], div_points[self.rank+1]

    def __len__(self) -> int:
        """
        Returns the total number of batches.

        :return: total number of batches
        """
        return len(self.batch_sampler)
