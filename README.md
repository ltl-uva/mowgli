# Mowgli

## Installation
* Python version >= 3.8
* To install mowgli and develop locally:
```bash
git clone git@github.com:ltl-uva/mowgli.git
cd mowgli/
pip install --editable ./ 
```

## Getting started
### How to train a model
* Before training, data needs to be pre-processed (e.g. using Moses) and a vocabulary needs to be created. See `scripts/build_vocab.py` for details on vocabulary creation.
* Training is done by pointing to a `yaml` file: `python -m mowgli train configs/${YOUR_CONFIG}.yaml`

### How to do inference
* Inference is done by pointing to a `yaml` file: `python -m mowgli test configs/${YOUR_CONFIG}.yaml`

## Developers
Mowgli is developed by [David Stap](https://davidstap.github.io) (University of Amsterdam).
