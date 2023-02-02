from mowgli.constants import BOS_TOKEN, EOS_TOKEN
import numpy as np
import bisect

import torch
from torch import LongTensor

class MowgliDataset(torch.utils.data.Dataset):
    """
    Base class for all datasets.
    """
    def get_only_stats(self, idx):
        return self.__getitem__(idx, get_sentences=False)

    def __getitem__(self, idx: int, get_sentences: bool = True) -> dict:
        """
        Returns index row from parallel or multiparallel dataset.

        :param idx: index
        :return: datapoint from indice, consisting of index, source token
            indices, target token indices, source length and target length
        """
        if   self.type == "parallel":       return self._get_parallel_item(idx, get_sentences)
        elif self.type == "multiparallel": return self._get_multiparallel_item(idx, get_sentences)

        else:                               raise NotImplementedError(f"Dataset type `{self.type}` is not implemented.")


    def _get_parallel_item(self, idx: int, get_sentences: bool = True) -> dict:
        """Return index row from parallel dataset. Embedding indices are computed on the fly."""
        item = {
            "idx":          self.data[idx]["idx"],
            "pair":         self.data[idx]["pair"],
            "src_len":      self.data[idx]["src_len"],
            "trg_len":      self.data[idx]["trg_len"],
            "src_lang":     self.data[idx]["src_lang"],
            "trg_lang":     self.data[idx]["trg_lang"],
        }

        if get_sentences:
            src_idx, trg_idx = [], []

            # Insert target language prefix
            if self.trg_tag_enc:
                src_idx.append(self.vocab[self.src_key].s2i[self.trg_tag])

            # Insert `begin of sentence` symbol for target
            trg_idx.append(self.vocab[self.trg_key].s2i[BOS_TOKEN])

            # Add embedding indices w/ vocab for all tokens
            tok_src = self.tokenizer(self.raw_data[self.data[idx]["idx"]][self.data[idx]["src_lang"]])
            tok_trg = self.tokenizer(self.raw_data[self.data[idx]["idx"]][self.data[idx]["trg_lang"]])
            src_idx.extend([self.vocab[self.src_key].s2i[t] for t in tok_src])
            trg_idx.extend([self.vocab[self.trg_key].s2i[t] for t in tok_trg])

            # Insert 'end of sentence' symbol for source and target
            src_idx.append(self.vocab[self.src_key].s2i[EOS_TOKEN])
            trg_idx.append(self.vocab[self.trg_key].s2i[EOS_TOKEN])

            item["src"] = np.array(src_idx)
            item["trg"] = np.array(trg_idx)

        return item


    def _get_multiparallel_item(self, idx: int, get_sentences: bool = True) -> dict:
        """Return index row from multiparallel dataset. Embedding indices are computed on the fly."""
        item = {
            "idx":          self.data[idx]["idx"],
            "src1_len":     self.data[idx]["src1_len"],
            "src2_len":     self.data[idx]["src2_len"],
            "src_len":      self.data[idx]["src_len"],
            "trg_len":      self.data[idx]["trg_len"],
            "src1_lang":    self.data[idx]["src1_lang"],
            "src2_lang":    self.data[idx]["src2_lang"],
            "trg_lang":     self.data[idx]["trg_lang"],
        }

        if get_sentences:
            src1_idx, src2_idx, trg_idx = [], [], []

            # Insert target language prefix
            if self.trg_tag_enc:
                src1_idx.append(self.vocab[self.src1_key].s2i[self.trg_tag])
                src2_idx.append(self.vocab[self.src2_key].s2i[self.trg_tag])

            # Insert 'begin of sentence' symbol for target
            trg_idx.append(self.vocab[self.trg_key].s2i[BOS_TOKEN])

            # Add embedding indices w/ vocab
            tok_src1 = self.tokenizer(self.raw_data[self.data[idx]["idx"]][self.data[idx]["src1_lang"]])
            tok_src2 = self.tokenizer(self.raw_data[self.data[idx]["idx"]][self.data[idx]["src2_lang"]])
            tok_trg  = self.tokenizer(self.raw_data[self.data[idx]["idx"]][self.data[idx]["trg_lang"]])
            src1_idx.extend([self.vocab[self.src1_key].s2i[t] for t in tok_src1])
            src2_idx.extend([self.vocab[self.src2_key].s2i[t] for t in tok_src2])
            trg_idx.extend( [self.vocab[self.trg_key ].s2i[t] for t in tok_trg])

            # insert 'end of sentence' symbol for source and target
            src1_idx.append(self.vocab[self.src1_key].s2i[EOS_TOKEN])
            src2_idx.append(self.vocab[self.src2_key].s2i[EOS_TOKEN])
            trg_idx.append( self.vocab[self.trg_key ].s2i[EOS_TOKEN])

            item["src1"] = np.array(src1_idx)
            item["src2"] = np.array(src2_idx)
            item["trg"]  = np.array(trg_idx)

        return item

    def __len__(self) -> int:
        """Returns length of dataset."""
        return len(self.data)

    @classmethod
    def splits(cls, **kwargs):
        """Return train, validation and test splits."""
        raise NotImplementedError


class MowgliConcatDataset(torch.utils.data.ConcatDataset):
    """
    Base class for concatenated datasets. Stores type of dataset.
    """
    def __init__(self, datasets: list):
        super().__init__(datasets)
        assert all([d.type == datasets[0].type for d in datasets])
        self.type = datasets[0].type
        self.total_tokens = sum([d.total_tokens for d in datasets])

        # Store graph
        # Assume graph is always the same (for different language combinations)
        assert len(set([d.graph for d in datasets])) == 1
        self.graph = datasets[0].graph


    def get_only_stats(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx].get_only_stats(sample_idx)
