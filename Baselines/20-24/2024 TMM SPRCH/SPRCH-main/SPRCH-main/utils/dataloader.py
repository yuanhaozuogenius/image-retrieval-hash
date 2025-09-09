# utils/dataloader.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100 as TV_CIFAR100
from PIL import Image


# ---- 顶层 OneHot（Windows 多进程可序列化） ----
class OneHot:
    def __init__(self, nclass: int):
        self.nclass = nclass
    def __call__(self, index):
        idx = torch.tensor(int(index)).long()
        return F.one_hot(idx, num_classes=self.nclass).float()


# ---- 通用 txt 列表数据集（非 CIFAR 使用） ----
class MyDataset(nn.Module):
    def __init__(self, data_path, data_name, txt, transform=None, target_transform=None):
        super().__init__()
        images, labels = [], []
        with open(txt, 'r') as fp:
            for line in fp:
                p, *ls = line.strip().split()
                images.append(p)
                labels.append([float(x) for x in ls])
        self.images = [os.path.join(data_path, data_name, p) for p in images]
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(self.labels[i])

    def __len__(self):
        return len(self.images)


def _tfms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    t_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm
    ])
    t_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), norm
    ])
    return t_train, t_eval


def _root_with(subdir: str, base: str) -> str:
    """若 base/subdir 存在则用之，否则用 base（兼容 D:\\Datasets\\cifar10\\... 的布局）"""
    cand = os.path.join(base, subdir)
    return cand if os.path.isdir(cand) else base


def _make_cifar10_full(data_path: str, batch_size: int):
    t_train, t_eval = _tfms()
    tgt = OneHot(10)
    root = _root_with('cifar10', data_path)  # 兼容 D:\Datasets\cifar10\cifar-10-batches-py
    trainset  = CIFAR10(root=root, train=True,  download=False, transform=t_train, target_transform=tgt)
    databases = CIFAR10(root=root, train=True,  download=False, transform=t_train, target_transform=tgt)
    testset   = CIFAR10(root=root, train=False, download=False, transform=t_eval,  target_transform=tgt)

    train_loader    = DataLoader(trainset,  batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    database_loader = DataLoader(databases, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader     = DataLoader(testset,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f'train set: {len(trainset)}', f'database set: {len(databases)}', f'test set: {len(testset)}')
    return train_loader, database_loader, test_loader


def _make_cifar100_full(data_path: str, batch_size: int):
    t_train, t_eval = _tfms()
    tgt = OneHot(100)
    root = _root_with('cifar100', data_path)  # 兼容 D:\Datasets\cifar100\cifar-100-python
    trainset  = TV_CIFAR100(root=root, train=True,  transform=t_train, target_transform=tgt)
    databases = TV_CIFAR100(root=root, train=True,  transform=t_train, target_transform=tgt)
    testset   = TV_CIFAR100(root=root, train=False, transform=t_eval,  target_transform=tgt)

    train_loader    = DataLoader(trainset,  batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True)
    database_loader = DataLoader(databases, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader     = DataLoader(testset,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print(f'train set: {len(trainset)}', f'database set: {len(databases)}', f'test set: {len(testset)}')
    return train_loader, database_loader, test_loader


def init_dataloader(data_path, data_name, train_list, database_list, test_list, batchSize):
    name = str(data_name).lower()
    if name in ['cifar10', 'cifar-10']:
        return _make_cifar10_full(data_path, batchSize)
    if name == 'cifar100':
        return _make_cifar100_full(data_path, batchSize)

    # 其他数据集：仍按 txt 列表
    t_train, t_eval = _tfms()
    train    = MyDataset(data_path, data_name, train_list,    transform=t_train)
    database = MyDataset(data_path, data_name, database_list, transform=t_eval)
    test     = MyDataset(data_path, data_name, test_list,     transform=t_eval)

    train_loader    = DataLoader(train,    batch_size=batchSize, shuffle=True,  num_workers=8, pin_memory=True)
    database_loader = DataLoader(database, batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=True)
    test_loader     = DataLoader(test,     batch_size=batchSize, shuffle=False, num_workers=8, pin_memory=True)

    print(f'train set: {len(train)}', f'database set: {len(database)}', f'test set: {len(test)}')
    return train_loader, database_loader, test_loader
