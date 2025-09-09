import os
import time
from time import strftime, localtime
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import scipy.io as io

from utils.evaluate import *
from utils.dataloader import *
from losses import *


def get_config():
    # —— 基础配置：可按需改默认值（无需命令行参数）——
    config = {
        "data_path": r"D:\Datasets",
        # "data_name": "cifar100",     # 在这里切换数据集：cifar10 / cifar100 / coco / voc2012 / ...
        "data_name": "cifar10",     # 在这里切换数据集：cifar10 / cifar100 / coco / voc2012 / ...
        "outf": "save",
        "checkpoint": 10,
        "batchSize": 64,
        "binary_bits": 64,
        "temp": 0.3,
        "epochs": 100,
        "lr": 1e-5,
        "loss": "p2p",
        "weighting": True,
        "self_paced": True,
        "device": torch.device("cuda:0"),
        "info": "[SPRCH]",
    }

    # —— 按数据集名自动设置 n_class 与 topK（并同步至 data_class / k）——
    name = config["data_name"].lower()
    if "cifar" in name:
        config["topK"] = -1
        config["n_class"] = 100 if "100" in name else 10
    elif name in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif name == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif name == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif name == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100  # 若你要 1000 类，改成 1000 即可
    elif name == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 38
    elif name in ["voc2012", "newvoc"]:
        config["topK"] = 1000
        config["n_class"] = 20
    else:
        # 默认兜底：不改写
        config.setdefault("topK", 5000)
        config.setdefault("n_class", config.get("data_class", 10))

    # 同步老字段
    config["data_class"] = config["n_class"]
    config["k"] = config["topK"]
    return config


def train(dataloader, net, optimizer, criterion, epoch, config, opt_for_loss, emb):
    accum_loss = 0.0
    net.train()
    for _, (img, label) in enumerate(dataloader):
        img = img.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        features = net(img)
        features = torch.tanh(features)

        # 原型向量：用 config 中的 data_class 保证与标签 one-hot 一致
        prototypes = emb(torch.eye(config["data_class"], device=label.device))
        prototypes = torch.tanh(prototypes)

        # 注意：传入 losses.py 的 opt 使用 SimpleNamespace，兼容老的属性访问写法
        loss = criterion(features, prototypes, label, epoch, opt_for_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
    return accum_loss / max(1, len(dataloader))


def train_val(config):
    print("config:", config)

    # —— 输出目录（全部确保为 str，避免 join 类型错误）——
    outf = os.path.join(
        str(config["outf"]),
        f'{config["loss"]}_{config["weighting"]}_{config["self_paced"]}',
        "SPRCH",
        str(config["data_name"]),
        str(config["binary_bits"]),
    )
    os.makedirs(outf, exist_ok=True)

    feed_random_seed()

    # ===== Dataset =====
    train_list = os.path.join("data", config["data_name"], "train.txt")
    database_list = os.path.join("data", config["data_name"], "database.txt")
    test_list = os.path.join("data", config["data_name"], "test.txt")
    train_loader, database_loader, test_loader = init_dataloader(
        config["data_path"], config["data_name"], train_list, database_list, test_list, config["batchSize"]
    )

    # ===== Model =====
    net = torchvision.models.resnet18(pretrained=True)
    net.fc = nn.Linear(512, config["binary_bits"])
    net.cuda()

    class Embedding(nn.Module):
        def __init__(self, n_class, n_bits):
            super().__init__()
            self.Embedding = nn.Linear(n_class, n_bits)
        def forward(self, x):
            return self.Embedding(x)

    emb = Embedding(config["data_class"], config["binary_bits"]).cuda()

    # ===== Loss / Optim =====
    criterion = SupConLoss(
        loss=config["loss"],
        temperature=config["temp"],
        data_class=config["data_class"]
    ).cuda()

    # 仅用于 losses.py（兼容属性访问）
    opt_for_loss = SimpleNamespace(**{k: v for k, v in config.items()})

    hash_id = list(map(id, net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in hash_id, net.parameters())
    optimizer = optim.Adam([
        {'params': feature_params, 'lr': config["lr"]},
        {'params': emb.parameters(), 'lr': 100 * config["lr"]},
        {'params': net.fc.parameters(), 'lr': 10 * config["lr"]}
    ])

    # ===== Train =====
    BestmAP = 0.0
    INFO_TAG = "[SPRCH]"  # 日志前缀按你的示例

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train(train_loader, net, optimizer, criterion, epoch, config, opt_for_loss, emb)
        now = strftime("%H:%M:%S", localtime())

        # 每个 epoch 打印（样式与示例一致）
        print(f"{INFO_TAG}[{epoch:03d}/{config['epochs']}][{now}] "
              f"bit:{config['binary_bits']}, dataset:{config['data_name']}, loss:{train_loss:.3f}")

        if epoch % config["checkpoint"] == 0:
            rB, rL = compute_result(database_loader, net)
            qB, qL = compute_result(test_loader, net)
            mAP = calc_topMap(qB, rB, qL, rL, config["k"])
            r_2 = calc_r_2(qB, rB, qL, rL)

            if BestmAP < mAP:
                BestmAP = mAP
                io.savemat(os.path.join(outf, 'save.mat'), {
                    'train_code': rB,
                    'L_tr': rL,
                    'test_code': qB,
                    'L_te': qL
                })
                torch.save(net.state_dict(), os.path.join(outf, 'save.pth'))

            print(f"{INFO_TAG} epoch:{epoch}, bit:{config['binary_bits']}, dataset:{config['data_name']}, "
                  f"MAP:{mAP:.3f}, Best MAP: {BestmAP:.3f}")


if __name__ == "__main__":
    config = get_config()
    train_val(config)
