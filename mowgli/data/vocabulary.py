"""
Vocabulary module
"""
from collections import defaultdict, Counter
from typing import List
import numpy as np

from mowgli.constants import (
    UNK_TOKEN,
    DEFAULT_UNK_ID,
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN
)


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(
        self,
        tokens: List[str] = None,
        file: str = None,
        specials = [],
    ) -> None:
        """
        Create vocabulary from list of tokens or file.

        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        :param tokens: list of tokens
        :param file: file to load vocabulary from
        """
        # warning: s2i grows with unknown tokens, don't use for saving or size

        # standard special symbols + additional special symbols
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + specials

        self.s2i = defaultdict(DEFAULT_UNK_ID)
        self.i2s = []
        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def _from_list(self, tokens: List[str] = None) -> None:
        """
        Make vocabulary from list of tokens.
        Tokens are assumed to be unique and pre-selected.
        Special symbols are added if not in list.

        :param tokens: list of tokens
        """
        self.add_tokens(tokens=self.specials+tokens)
        assert len(self.s2i) == len(self.i2s)

    def _from_file(self, file: str) -> None:
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: path to file where the vocabulary is loaded from
        """
        tokens = []
        with open(file, "r") as open_file:
            for line in open_file:
                tokens.append(line.strip("\n"))
        self._from_list(tokens)

    def __str__(self) -> str:
        return self.s2i.__str__()

    def to_file(self, file: str) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        with open(file, "w") as open_file:
            for t in self.i2s:
                open_file.write("{}\n".format(t))

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add list of tokens to vocabulary

        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self.i2s)
            # add to vocab if not already there
            if t not in self.i2s:
                self.i2s.append(t)
                self.s2i[t] = new_index

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.s2i[token] == DEFAULT_UNK_ID()

    def __len__(self) -> int:
        return len(self.i2s)

    def array_to_sentence(
        self,
        array: np.array,
        cut_at_eos=True,
        skip_pad=True
    ) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self.i2s[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            if skip_pad and s == PAD_TOKEN:
                continue
            sentence.append(s)
        return sentence

    def arrays_to_sentences(
        self,
        arrays: np.array,
        cut_at_eos=True,
        skip_pad=True
    ) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(
                self.array_to_sentence(
                    array=array, cut_at_eos=cut_at_eos, skip_pad=skip_pad)
                )
        return sentences
