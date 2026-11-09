"""
Training ResNet-20 on different datasets
python src/train.py svhn --epochs 15
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as D 


# Transform into 3x32x32
def adapt_transform(d, train=False):
    steps = []
    if d.in_channels == 1:
        steps.append(transforms.Grayscale(3))
    if d.image_size != 32:
        steps.append(transforms.Resize(32))
    if train:
        steps.append(transforms.RandomCrop(32, padding=4)) 

    mean = d.mean * (3 if len(d.mean) == 1 else 1)
    std = d.std * (3 if len(d.std) == 1 else 1)
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


# Train and test loaders
def build_loaders(d, batch_size=128, num_workers=2):
    def loader(train):
        kw = dict(root=D.DATA_ROOT, download=True, transform=adapt_transform(d, train), **d.split_kwargs)
        if d.name == "svhn":
            kw["split"] = "train" if train else "test"
        else:
            kw["train"] = train
        return torch.utils.data.DataLoader(d.torchvision_cls(**kw), batch_size=batch_size,
                                           shuffle=train, num_workers=num_workers)
    return loader(True), loader(False)


def build_model(n_classes, device=D.DEVICE, pretrained=True):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=pretrained)
    if n_classes != 10:
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


# Accuracy in %
# offset 1 in EMNIST-letters 
@torch.no_grad()
def evaluate(model, loader, offset=0, device=D.DEVICE):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(1) == (y - offset).to(device)).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


# Fine-tuning OneCycle, keeps the checkpoint of the best epoch
def train(name, epochs=15, lr=0.05, batch_size=128, num_workers=2, device=D.DEVICE):
    d = D.get_dataset(name)
    train_loader, test_loader = build_loaders(d, batch_size, num_workers)
    offset = d.label_offset
    model = build_model(d.n_classes, device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=epochs, steps_per_epoch=len(train_loader))

    best, t0 = 0.0, time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(device)), (y - offset).to(device))
            loss.backward()
            opt.step()
            sched.step()

        # Save only if improvement
        acc = evaluate(model, test_loader, offset, device)
        if acc > best:
            best = acc
            os.makedirs(D.CKPT_ROOT, exist_ok=True)
            torch.save(model.state_dict(), d.checkpoint_path())
        print(f"epoch {ep:>3}/{epochs}  test {acc:.2f}%  (best {best:.2f}%)  [{time.time() - t0:.0f}s]", flush=True)
    return model, best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=list(D.DATASETS))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    train(a.dataset, a.epochs, a.lr, a.batch_size, a.workers)
