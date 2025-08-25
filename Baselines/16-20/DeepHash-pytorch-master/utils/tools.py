import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
import os
import re
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets

"""
根据数据集名称设置分类数、topK 评价范围和数据路径等信息。
"""
def config_dataset(config):
    base_path = "./data/"  # 相对路径，基于项目根目录

    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 38
    elif config["dataset"] in ["voc2012", "newvoc"]:
        config["topK"] = 1000
        config["n_class"] = 20

    config["data_path"] = base_path + config["dataset"] + "/"

    config["data"] = {
        "train_set": {"list_path": base_path + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": base_path + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": base_path + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}
    }

    return config



# 预定义的检索样本数量范围，用于绘制 PR 曲线
draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

"""
根据查询集与数据库的特征和标签计算 Precision-Recall 曲线数据。
"""
def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R


"""
自定义图像数据集类，从 txt 列表中读取图像路径与多标签数据。
"""
class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        # 返回处理后的图像 tensor，标签（向量），样本索引（有时用于追踪或采样）
        return img, target, index

    def __len__(self):
        return len(self.imgs)

"""
定义图像预处理操作，包括 Resize、Crop、ToTensor 和 Normalize。
根据是 train_set 还是 test/database 设置不同的变换。
"""
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])

"""
CIFAR10 数据集的自定义版本，将标签转为 one-hot，并进行 transform。
"""
class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

"""
处理 CIFAR10 类数据集，按照固定比例划分 train、test 和 database 并返回 DataLoader。
"""
def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = 'dataset/cifar/'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

"""
根据 config 配置，加载非 CIFAR 的通用图像数据集并返回对应的 DataLoader。
"""
def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])



"""
清理 save 目录中除最佳 mAP 文件外的所有模型/中间文件
"""
def clean_save_dir_keep_best(save_dir, dataset_name):
    if not os.path.exists(save_dir):
        print(f"[clean_save_dir_keep_best] ⚠️ 目录不存在: {save_dir}，跳过清理。")
        return

    # 支持 dataset_tag-<10位浮点数>-xxx
    pattern = re.compile(r"^(.+)-(\d+\.\d{10})-")
    file_groups = {}  # {score: [file1, file2, ...]}

    for filename in os.listdir(save_dir):
        match = pattern.match(filename)
        if match:
            score = float(match.group(2))
            file_path = os.path.join(save_dir, filename)
            file_groups.setdefault(score, []).append(file_path)

    if not file_groups:
        print("[clean_save_dir_keep_best] ❌ No matching files to clean.")
        return

    best_score = max(file_groups.keys())
    print(f"[clean_save_dir_keep_best] ✅ Keep mAP={best_score:.10f}, delete other {len(file_groups) - 1} groups.")

    deleted = 0
    for score, files in file_groups.items():
        if score != best_score:
            for file_path in files:
                try:
                    os.remove(file_path)
                    deleted += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"[clean_save_dir_keep_best] 🧹 Deleted {deleted} files.")


"""
通过模型提取图像的哈希特征向量（sign）与对应标签，并返回全部结果。
"""
def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

"""
计算两个哈希码矩阵之间的汉明距离。
"""
def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

"""
计算 mAP（mean Average Precision）指标，衡量哈希检索质量。
rB	retrieval hash code（数据库图像的哈希码），shape: [N_database, n_bits]
qB	query hash code（查询图像的哈希码），shape: [N_query, n_bits]
retrievalL	数据库图像的标签（multi-hot），shape: [N_database, n_class]
queryL	查询图像的标签，shape: [N_query, n_class]
topk	指定从数据库中检索的前 topk 个样本用于评估

返回的是一个浮点数 topkmap，表示：
在前 topk 个检索结果中，平均每个查询的检索精度均值（mean Average Precision）
即：整体系统的平均检索性能指标（越高越好，最大为 1.0）


以每个查询为例：
计算查询样本与所有数据库样本的标签是否有重合（ground truth）；
计算它与所有数据库样本的汉明距离 CalcHammingDist；
排序后，取前 topk 个排序结果；
计算这些结果中 relevant 的平均精度；
对所有查询样本做平均，得到最终 topkmap。
"""
def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
