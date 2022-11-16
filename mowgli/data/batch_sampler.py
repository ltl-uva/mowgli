from contextlib import contextmanager
import copy
import math
import random
from typing import Callable, Generator, Optional

import torch

from mowgli.data import LogosBatchSampler


class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return copy.deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))


class BatchSampler(LogosBatchSampler):
    """
    Batch sampler for parallel and multiparallel datasets. Creates an iterable
    that is used by torch.Dataloader.
    """
    def __init__(
        self,
        dataset: torch.utils.data.ConcatDataset,
        batch_size: int,
        sort_key: Callable[[dict], int],
        batch_size_fn = lambda new, count, sofar: count,
        train: Optional[bool] = True,
        shuffle: Optional[bool] = None,
    ):
        """
        Initializes batch sampler

        :param dataset: dataset for which iterator will be created
        :param batch_size: batch size
        :param sort_key: sorting function
        :param batch_size_fn: function to calculate batch size
        :param train: whether this is a training dataset
        :param repeat: whether to repeat epoch
        :param shuffle: whether to shuffle the dataset
        """
        self.batch_size = batch_size
        self.train = train
        self.dataset = dataset
        self.batch_size_fn = batch_size_fn
        self.shuffle = train if shuffle is None else shuffle
        self.sort_key = sort_key
        self.random_shuffler = RandomShuffler()

    def data(self) -> list:
        """
        Return the examples in the dataset in order, sorted, or shuffled.

        :return: processed dataset
        """
        # Store (shuffled) indices in memory
        if self.shuffle:
            # return shuffled indices
            return self.random_shuffler(range(len(self.dataset)))

        else:
            # we can afford to load the whole validation / test set into memory
            if not self.train:
                return [(i, self.dataset.get_only_stats(i)) for i in range(len(self.dataset))]

            # return indices
            return list(range(len(self.dataset)))

    def init_epoch(self) -> None:
        """Sets up the batch generator for a new epoch."""
        self._random_state_this_epoch = self.random_shuffler.random_state
        self.create_batches()

    def create_batches(self):
        if not self.train:
            self.batches = self.batch(self.data(), self.batch_size, self.batch_size_fn)
        else:
            self.batches = self.bucketize(self.data())

    def bucketize(self, data: list) -> Generator[list, None, None]:
        """
        Partitions data into buckets of size 100*`batch_size`. In this case,
        batch size is #sentences, not tokens, even if `self.batch_size_fn`
        is specified otherwise). This ensures a large enough search space.

        :param data: to be bucketed data
        :return: batch of sorted examples
        """
        for bucket in self.batch(data, self.batch_size*100, lambda new, count, sofar: count):
            # Sort examples within each chunk using sort_key
            idx = [(i, self.sort_key(self.dataset.get_only_stats(i))) for i in bucket]
            idx = sorted(idx, key=lambda x: x[1])
            sorted_result = [(i[0], self.dataset.get_only_stats(i[0])) for i in idx]

            # Batch these sorted examples
            for b in self.random_shuffler(list(self.batch(sorted_result, self.batch_size, self.batch_size_fn))):
                yield b


    def __len__(self) -> int:
        """Returns number of batches in epoch."""
        if self.batch_size_fn is not None:
            raise NotImplementedError
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Generator[list, None, None]:
        """
        Yields a batch, i.e. a list of dataset indices.

        :return: single (next) batch
        """
        while True:
            self.init_epoch()

            for idx, minibatch in enumerate(self.batches):
                # Yield indices, which are transformed to batches by collator
                yield [x[0] for x in minibatch]

            # return after finishing epoch
            return


    def batch(self, data: list, batch_size: int, batch_size_fn: Callable[[int, int, int], int]) -> Generator[list, None, None]:
        """
        Yield elements from data in chunks of `batch_size`.

        :param data: list of dataset indices
        :param batch_size: batch size
        :param batch_size_fn: function to calculate batch size
        :return: filled batch
        """
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
        if minibatch:
            yield minibatch
