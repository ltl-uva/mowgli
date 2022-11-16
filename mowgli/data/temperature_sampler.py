from collections import defaultdict
import random


class TemperatureSampler:
    def __init__(self, dataset, temp):
        """
        """
        self.dataset = dataset
        self.idx_for_dataset = self._get_idx_for_dataset()
        self.dataset_sizes = {
            pair: len(idx) for pair, idx in self.idx_for_dataset.items()
        }

        # Calculate sample probabilities based on temperature sampling formula
        total_size = sum(self.dataset_sizes.values())
        sample_probs = {
            pair: (size / total_size) ** (1.0 / temp)
            for pair, size in self.dataset_sizes.items()
        }
        sample_probs = {
            pair: p / sum(sample_probs.values())
            for pair, p in sample_probs.items()
        }

        # Calculate sizes after sampling for all pairs
        self.new_dataset_sizes = {
            pair: int(p*total_size) for pair, p in sample_probs.items()
        }

    def sample(self):
        """
        """
        idx_for_epoch = []
        for pair in self.dataset_sizes.keys():

            # Upsample if size increased, leave unchanged if size is equal
            if self.dataset_sizes[pair] <= self.new_dataset_sizes[pair]:
                repeat, diff = divmod(self.new_dataset_sizes[pair], self.dataset_sizes[pair])
                for _ in range(repeat):
                    idx_for_epoch += self.idx_for_dataset[pair]

                idx_for_epoch += random.sample(
                    self.idx_for_dataset[pair], diff
                )

            # Downsample if size decreased
            elif self.dataset_sizes[pair] > self.new_dataset_sizes[pair]:
                idx_for_epoch += random.sample(
                    self.idx_for_dataset[pair], self.new_dataset_sizes[pair]
                )

        return [self.dataset[idx] for idx in idx_for_epoch]


    def _get_idx_for_dataset(self):
        """
        """
        idx_for_pair = defaultdict(list)

        for idx, x in enumerate(self.dataset):
            idx_for_pair[x["pair"]].append(idx)

        return idx_for_pair
