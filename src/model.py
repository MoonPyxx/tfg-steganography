"""
python src/model.py [hub_model_name]
"""

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms

DEFAULT_MODEL = "cifar10_resnet20"
MODEL_NAME = DEFAULT_MODEL 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "data")
MEAN, STD = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]


def get_model(name=DEFAULT_MODEL):
    return torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True).to(DEVICE)


# CIFAR-10 
def get_test_loader():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=256, num_workers=2)


# Top-1 accuracy
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def inventory(model):
    rows = sorted(((n, p.numel()) for n, p in model.named_parameters()), key=lambda r: -r[1])
    for n, k in rows:
        print(f"{n:<42} {k:>10,}")
    print(f"TOTAL {sum(k for _, k in rows):,} parameters")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model = get_model(name)
    print(f"baseline accuracy: {evaluate(model, get_test_loader()):.2f}%")
    inventory(model)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{name}_clean.pt")
