import os

import torch
import torchvision
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
CKPT_ROOT = os.path.join(REPO_ROOT, "checkpoints")


class Dataset:
    def __init__(self, name, torchvision_cls, mean, std, n_classes, hub_model=None,
                 in_channels=3, image_size=32, split_kwargs=None, label_offset=0):
        self.name = name
        self.torchvision_cls = torchvision_cls
        self.mean = mean
        self.std = std
        self.n_classes = n_classes
        self.hub_model = hub_model
        self.in_channels = in_channels
        self.image_size = image_size
        self.split_kwargs = split_kwargs or {}
        self.label_offset = label_offset  # 1 for EMNIST-letters (labels 1..26)

    # pretrained model torch.hub or train.py
    @property
    def needs_training(self):
        return self.hub_model is None

    # torchvision dataset with plain normalization
    def _build(self, train):
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        kw = dict(root=DATA_ROOT, download=True, **self.split_kwargs, transform=tf)
        if self.torchvision_cls is torchvision.datasets.SVHN:
            kw["split"] = "train" if train else "test"
        else:
            kw["train"] = train
        return self.torchvision_cls(**kw)

    def test_loader(self, batch_size=BATCH_SIZE, num_workers=2):
        return torch.utils.data.DataLoader(self._build(False), batch_size=batch_size, num_workers=num_workers)

    def train_loader(self, batch_size=BATCH_SIZE, num_workers=2):
        return torch.utils.data.DataLoader(self._build(True), batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers)

    # Pretrained model from torch.hub 
    def load_model(self, device=DEVICE):
        return torch.hub.load("chenyaofo/pytorch-cifar-models", self.hub_model, pretrained=True).to(device)

    def checkpoint_path(self, suffix="clean"):
        return os.path.join(CKPT_ROOT, f"{self.name}_resnet20_{suffix}.pt")

    def __repr__(self):
        return f"<Dataset {self.name}: {self.n_classes} classes, {self.hub_model or 'TRAIN'}>"


DATASETS = {
    "cifar10": Dataset("cifar10", torchvision.datasets.CIFAR10,
                       [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010], 10, "cifar10_resnet20"),
    "cifar100": Dataset("cifar100", torchvision.datasets.CIFAR100,
                        [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762], 100, "cifar100_resnet20"),
    "svhn": Dataset("svhn", torchvision.datasets.SVHN,
                    [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970], 10),
    "fashion_mnist": Dataset("fashion_mnist", torchvision.datasets.FashionMNIST,
                             [0.2860], [0.3530], 10, in_channels=1, image_size=28),
    "emnist": Dataset("emnist", torchvision.datasets.EMNIST,
                      [0.1751], [0.3332], 26, in_channels=1, image_size=28, split_kwargs={"split": "letters"},
                      label_offset=1),
}


def get_dataset(name):
    return DATASETS[name]


# Datasets with a pretrained model
def ready():
    return [d for d in DATASETS.values() if not d.needs_training]


# Datasets that need train.py
def pending():
    return [d for d in DATASETS.values() if d.needs_training]
