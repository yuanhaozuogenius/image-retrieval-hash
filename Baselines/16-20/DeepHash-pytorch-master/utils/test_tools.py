import os
import torch
from network import *

import torch.optim as optim
import numpy as np

from tools import (
    config_dataset, ImageList, image_transform,
    CalcHammingDist, pr_curve
)

torch.multiprocessing.set_sharing_strategy('file_system')

# 导入 precision_recall_curve.py 触发绘图
import precision_recall_curve as prc


# 测试数据集路径
def verify_dataset_path(dataset_name):
    # 将 base_path 设置为项目根目录下的 data 目录
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    data_path = os.path.join(base_path, dataset_name)
    train_file = os.path.join(data_path, "train.txt")
    database_file = os.path.join(data_path, "database.txt")
    test_file = os.path.join(data_path, "test.txt")

    print("=======================================")
    print(f"Checking dataset: {dataset_name}")
    print("---------------------------------------")
    print(f"Train file exists:    {os.path.exists(train_file)}")
    print(f"Database file exists: {os.path.exists(database_file)}")
    print(f"Test file exists:     {os.path.exists(test_file)}")
    print("=======================================\n")


# 测试 config_dataset
def test_config_dataset():
    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DHN]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 128,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 90,
        "test_map": 15,
        "save_path": "save/DHN",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [48],
    }
    config = config_dataset(config)
    print("Config Dataset Test:")
    print(config)
    print()


# 测试 ImageList
def test_image_list():
    dataset_name = "nuswide_21"
    base_path = "./data/"
    list_path = os.path.join(base_path, dataset_name, "train.txt")

    with open(list_path, "r") as f:
        image_list = f.readlines()

    dataset = ImageList(base_path + dataset_name + "/", image_list, transform=image_transform(256, 224, "train_set"))
    print(f"ImageList length: {len(dataset)}")
    img, target, idx = dataset[0]
    print(f"Sample image shape: {img.shape}, Target: {target}, Index: {idx}")
    print()


# 测试汉明距离
def test_hamming_distance():
    B1 = np.sign(np.random.randn(5, 8))
    B2 = np.sign(np.random.randn(10, 8))
    dist = CalcHammingDist(B1, B2)
    print("Hamming Distance Test:")
    print(dist)
    print()


# 测试 PR 曲线
def test_pr_curve():
    rF = np.sign(np.random.randn(10, 8))
    qF = np.sign(np.random.randn(3, 8))
    rL = np.eye(10)[np.random.randint(0, 10, 10)]
    qL = np.eye(10)[np.random.randint(0, 10, 3)]
    P, R = pr_curve(rF, qF, rL, qL, draw_range=[1, 5, 10])
    print("PR Curve Test:")
    print("Precision:", P)
    print("Recall:", R)
    print()


# 测试 precision_recall_curve.py 生成图像
def test_precision_recall_curve():
    print("Testing precision_recall_curve.py...")
    output_file = "pr.png"
    if os.path.exists(output_file):
        print(f"PR Curve image '{output_file}' generated successfully.")
    else:
        print(f"Failed to generate PR Curve image '{output_file}'.")
    print()


# 主测试入口
if __name__ == "__main__":
    # GPU测试
    print("是否检测到 GPU:", torch.cuda.is_available())
    print("当前默认设备:", torch.cuda.current_device() if torch.cuda.is_available() else "无 GPU")
    if torch.cuda.is_available():
        print("GPU 名称:", torch.cuda.get_device_name(0))
        print("已使用内存: %.2f MB" % (torch.cuda.memory_allocated() / 1024 / 1024))
        print("缓存占用: %.2f MB" % (torch.cuda.memory_reserved() / 1024 / 1024))

    # 函数功能测试
    print("=== Verify Dataset Path Test ===")
    verify_dataset_path("nuswide_21")
    # verify_dataset_path("coco")
    # verify_dataset_path("voc2012")

    print("=== Config Dataset Function Test ===")
    test_config_dataset()

    print("=== ImageList Class Test ===")
    test_image_list()

    print("=== Hamming Distance Function Test ===")
    test_hamming_distance()

    print("=== PR Curve Function Test ===")
    test_pr_curve()

    print("=== Precision-Recall Curve Plot Test ===")
    test_precision_recall_curve()
