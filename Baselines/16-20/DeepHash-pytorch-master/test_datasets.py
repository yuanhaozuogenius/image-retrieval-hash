"""
测试 CSQ_BLIP 系列各数据集加载是否正常。
复用 utils.tools 中的 config_dataset 和 get_data。
"""
import os
import torch
from utils.tools import config_dataset, get_data


def show_loader_info(loader, name):
    """ fetch 真实路径中的images
        打印DataLoader 基本信息与样本结构"""
    print(f"\n Testing loader: {name}")
    print(f" Total batches: {len(loader)}")

    # 取第一个 batch 看一下样本结构
    for imgs, labels, idxs, paths in loader:
        print(f"📦 Batch sample:")
        print(f"  - Image tensor shape: {tuple(imgs.shape)}")
        print(f"  - Label tensor shape: {tuple(labels.shape)}")
        print(f"  - 打印前3个样本信息:")
        for i in range(min(3, len(paths))):
            file_name = os.path.basename(paths[i])
            label_nonzero = torch.nonzero(labels[i]).squeeze().tolist()
            if isinstance(label_nonzero, int):
                label_nonzero = [label_nonzero]
            print(f"    [{i}]")
            print(f"      • File: {file_name}")
            print(f"      • Full path: {paths[i]}")
            print(f"      • Label indices: {label_nonzero}")
        break


def test_dataset(dataset_name):
    """通用测试函数：配置 -> 加载 -> 打印 txt文件的配置"""
    print(f"\n==============================")
    print(f"🧩 Testing dataset: {dataset_name}")
    print(f"==============================")

    config = {
        "dataset": dataset_name,
        "batch_size": 4,
        "resize_size": 256,
        "crop_size": 224,
    }
    subdir = DATASET_IMAGE_ROOTS.get(dataset_name, dataset_name)
    config["image_root"] = os.path.join(DATA_ROOT, subdir)
    config = config_dataset(config)
    train_loader, test_loader, db_loader, n_train, n_test, n_database = get_data(config)

    # 展示各个 DataLoader 的样本结构
    show_loader_info(train_loader, f"{dataset_name} (train_set)")
    show_loader_info(test_loader, f"{dataset_name} (test)")
    show_loader_info(db_loader, f"{dataset_name} (database)")


if __name__ == "__main__":
    DATA_ROOT = r"D:/Datasets"
    # 逻辑名 -> 实际图片目录
    DATASET_IMAGE_ROOTS = {
        "coco": "coco2017",
        "cifar10": "cifar10-image",
        "imagenet": "ImageNet100",
    }
    # 这里数据集名称需填写真实完整名称
    test_dataset("coco")
    test_dataset("cifar10")
    # test_dataset("ImageNet100")
