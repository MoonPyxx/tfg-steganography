# Steganography in neural network parameters

Code for TFG project on hiding an arbitrary payload in the weights of
ResNet-20 and measuring **capacity**, **fidelity** and **robustness** on five
datasets (CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST, EMNIST). It implements the
following techniques: LSB substitution, value mapping, sign mapping and
resilience training.

## Usage

```bash
python src/experiment_5datasets.py   # (LSB) capacity vs fidelity, noise
python src/experiment_methods.py     # LSB vs value vs sign mapping
python src/resilience.py cifar10     # resilience training
python src/train.py svhn             # train a model
```

Datasets are downloaded to `data/` on first run.

## Layout

```
src/          codecs, experiments and training
checkpoints/  clean and resilience-trained weights of the 5 models
results/   measured results (JSON)
```
