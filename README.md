# Mowgli
Mowgli is a lightweight neural machine translation framework based on the Transformer network.

## Installation
* Python version >= 3.8
* To install mowgli and develop locally:
```bash
# download code
git clone git@github.com:ltl-uva/mowgli.git
cd mowgli/

# create a virtual environment, for example using conda
conda create --name mowgli python==3.8
conda activate mowgli

# install
pip install --editable ./ 
```

## Getting started
Configuration is done through `yaml` files. See [configuration folder](https://github.com/ltl-uva/mowgli/tree/main/configs) for a list of all options.

### How to train a model (`mowgli train`)
* Before training, data needs to be pre-processed (e.g. using Moses) and a vocabulary needs to be created. See `scripts/build_vocab.py` for details on vocabulary creation.
* Training is done by pointing to a `yaml` file: `python -m mowgli train configs/${YOUR_CONFIG}.yaml`

### How to do inference (`mowgli test`)
* Inference is done by pointing to a `yaml` file: `python -m mowgli test configs/${YOUR_CONFIG}.yaml`

## Developers
Mowgli is developed by [David Stap](https://davidstap.github.io) (University of Amsterdam).

We take inspiration from other sequence-to-sequence frameworks such as [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor), [fairseq](https://github.com/facebookresearch/fairseq), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) and [JoeyNMT](https://github.com/joeynmt/joeynmt).

## Reference
If you use mowgli, please cite the following [paper]([https://arxiv.org/abs/1907.12484](https://aclanthology.org/2023.findings-emnlp.998/)):

```
@inproceedings{stap-etal-2023-viewing,
    title = "Viewing Knowledge Transfer in Multilingual Machine Translation Through a Representational Lens",
    author = "Stap, David  and
      Niculae, Vlad  and
      Monz, Christof",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.998",
    pages = "14973--14987",
}
```

