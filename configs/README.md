# YAML config options

## `data`
TODO

Name | Optional | Default | Description
--------- | :----------: | :-- | --:
`src`|no|NA|List of source languages for training.
`trg`|no|NA|List of target languages for training.
`valid_src`|yes|`src`|List of source languages for validation. Useful for validating on a subset of languages.
`valid_trg`|yes|`trg`|List of target languages for validation. Useful for validating on a subset of languages.
`test_src`|yes|`src`|List of source languages for testing. Useful for testing on a subset of languages.
`test_trg`|yes|`trg`|List of target languages for testing. Useful for testing on a subset of languages.
`train_path`|no|NA|Path to training data folder plus prefix. For example, we can have two files: `data/train.en` and `data/train.de`. In this case `train_path` is `data/train`.
`valid_path`|no|NA|Path to validation data folder plus prefix.
`test_path`|no|NA|Path to test data folder plus prefix.
`level`|no|NA|test
`lowercase`|no|NA|test
`max_sent_length`|no|NA|test
`src_voc_limit`|no|NA|test
`trg_voc_limit`|no|NA|test
`src_voc_min_freq`|no|NA|test
`trg_voc_min_freq`|no|NA|test
`share_vocab`|no|NA|test
`vocab`|no|NA|test
`parallel`|no|True|test
`multiparallel`|no|False|test
`num_workers`|no|8|test

centric


## `training`
TODO

## `testing`
TODO

## `model`
TODO
