# YAML config options

## `data`

Name | Optional | Default | Description
--------- | :----------: | :-- | --:
`src`|no|NA|list of source languages for training.
`trg`|no|NA|list of target languages for training.
`centric`|yes|`None`|only consider language directions that involve centric language.
`valid_src`|yes|`src`|list of source languages for validation. useful for validating on a subset of languages.
`valid_trg`|yes|`trg`|list of target languages for validation. useful for validating on a subset of languages.
`test_src`|yes|`src`|list of source languages for testing. useful for testing on a subset of languages.
`test_trg`|yes|`trg`|list of target languages for testing. useful for testing on a subset of languages.
`train_path`|no|NA|path to training data folder plus prefix. for example, we can have two files: `data/train.en` and `data/train.de`. in this case `train_path` is `data/train`.
`valid_path`|no|NA|path to validation data folder plus prefix.
`test_path`|no|NA|path to test data folder plus prefix.
`level`|no|NA|can be `word`, `bpe` or `char`.
`max_sent_length`|no|NA|maximum number of words in a sentence.
`src_voc_limit`|yes|`sys.maxsize`|source vocabulary contains this many most frequent tokens.
`trg_voc_limit`|yes|`sys.maxsize`|target vocabulary contains this many most frequent tokens.
`src_voc_min_freq`|yes|1|minimum source token frequency
`trg_voc_min_freq`|yes|1|minimum target token frequency
`share_vocab`|no|NA|whether to share the vocabulary between source and target; either `True` or `False`.
`vocab`|no|NA|path to vocabulary file. expects as suffix either `.shared` or `.${LANGUAGE}`.
`parallel`|no|True|whether to sample parallel batches.
`multiparallel`|no|False|whether to sample multi-parallel batches.
`num_workers`|no|8|how many sub-processes to use for data loading.

## `training`
TODO

## `testing`
Name | Optional | Default | Description
--------- | :----------: | :-- | --:
`beam_size`|yes|1|beam size in beam search.
`alpha`|yes|-1|length penalty for beam search (-1 for no penalty).
`postprocess`|yes|`True`|whether to remove BPE splits
`bpe_type`|yes|`subword-nmt`|bpe type (either `subword-nmt` or `sentencepiece`).
`sacrebleu`:`remove_whitespace`|yes|`True`|`sacrebleu` option to remove whitespace.
`sacrebleu`:`tokenize`|yes|`13a`|`sacrebleu` option for tokenization scheme.


## `model`
TODO
