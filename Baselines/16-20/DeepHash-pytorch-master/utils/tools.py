import math

import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
import os
import re
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import json
import time
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import spacy
import gc

"""
根据数据集名称设置分类数、topK 评价范围和数据路径等信息。
"""


def config_dataset(config):
    base_path = "./data/"  # 相对路径，基于项目根目录

    if "cifar" in config["dataset"]:
        config["topK"] = 5000
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
    elif config["dataset"] == "ImageNet100":
        config["topK"] = 5000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 38
    elif config["dataset"] in ["voc2012", "newvoc"]:
        config["topK"] = 5000
        config["n_class"] = 20

    config["data_path"] = base_path + config["dataset"] + "/"
    # 给通用数据集一个统一的图像根目录，默认为 data_path, 可以在外部通过 config["image_root"] 覆盖为绝对路径（如 D:/Datasets/...）
    if "image_root" not in config:
        config["image_root"] = config["data_path"]

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
        self.imgs = []
        self.transform = transform

        for raw in image_list:
            line = raw.strip()
            if not line:
                continue

            parts = line.split()  # 兼容空格或\t
            rel = parts[0].replace("\\", "/")  # 统一斜杠
            # ✅ 关键：绝对路径则直接用；否则与 data_path join
            if os.path.isabs(rel):
                full = rel
            else:
                full = os.path.normpath(os.path.join(data_path, rel))

            # 多标签（空格分隔）→ int 向量；单标签也兼容
            labels = np.array([int(x) for x in parts[1:]], dtype=np.int8)
            self.imgs.append((full, labels))

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        # 返回处理后的图像 tensor，标签（向量），样本索引（有时用于追踪或采样）
        return img, target, index, path

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
    cifar_dir = config["cifar10_dir"]

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
    # cifar_dataset_root = 'dataset/cifar/'

    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dir,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dir,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dir,
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
    elif config["dataset"] == "cifar10":
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
                                               # shuffle=True, 在txt文本中已经做了shuffle
                                               shuffle=False,
                                               num_workers=16)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=16)

    return train_loader, test_loader, database_loader, \
        train_index.shape[0], test_index.shape[0], database_index.shape[0]


"""
根据 config 配置，加载非 CIFAR 的通用图像数据集并返回对应的 DataLoader。
"""


def get_data(config):
    # if "cifar" in config["dataset"]:
    #     return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    # 统一图像根目录（可为绝对路径；也能走相对路径）
    image_root = config.get("image_root", config["data_path"])

    # _image_root = config.get("cifar10_dir", config["data_path"])
    for data_set in ["train_set", "test", "database"]:
        with open(data_config[data_set]["list_path"], "r") as f:
            lines = f.readlines()
        dsets[data_set] = ImageList(
            image_root,
            lines,
            transform=image_transform(config["resize_size"], config["crop_size"], data_set)
        )
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(
            dsets[data_set],
            batch_size=data_config[data_set]["batch_size"],
            shuffle=False, num_workers=16  # 改为False，在txt文本中已经做了shuffle
        )

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
        len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


# add 1113 构建 caption 专用 DataLoader（使用 train_all.txt） ======
def build_caption_loader(config):
    """
    功能：根据 train_all.txt 构建一个独立 DataLoader，用于 precompute_class_captions。
    参数：
        all_train_data : 训练全集的 txt（train_all.txt）
    """
    list_path = config["all_train_data"]
    image_root = config.get("image_root")
    resize_size = config["resize_size"]
    crop_size = config["crop_size"]
    batch_size = config["batch_size"]

    # 读取 txt
    with open(list_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 构造数据集
    dataset = ImageList(
        image_root,
        lines,
        transform=image_transform(resize_size, crop_size, "train_set")
    )
    print("load all train data for generating captions: len", len(dataset))
    # 构造 loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16
    )
    return loader


def load_class_captions(jsonl_path: str) -> Dict[int, List[str]]:
    """
    从 JSONL 缓存文件中加载每个类别的 captions。

    • 文件结构：每行一个 JSON 对象，形如：
    {"class": 3, "image_ids": [...], "captions": [{"text": "..."} , ...]}

    • 返回值：{class_id: [caption_text1, caption_text2, ...]}
      - 若文件不存在：自动创建空文件并返回空字典；
      - 若文件存在但无有效数据：返回空字典；
      - 存在且正常：读取
    """
    # 确保目录存在
    jsonl_dir = os.path.dirname(jsonl_path)
    os.makedirs(jsonl_dir, exist_ok=True)
    # 若文件不存在：创建空文件 + 打印提示
    if not os.path.isfile(jsonl_path):
        open(jsonl_path, "a", encoding="utf-8").close()
        print(f"[caption-cache] init empty cache at: {jsonl_path}")
        return {}
    # 存在且正常：读取
    caps: Dict[int, List[str]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                c = obj.get("class")
                arr = obj.get("captions")
                if c is None or not isinstance(arr, list) or not arr:
                    continue

                for item in arr:
                    if isinstance(item, dict):
                        t = str(item.get("text", "")).strip()
                        if t:
                            caps.setdefault(int(c), []).append(t)
            except Exception:
                pass
    return caps


def append_class_captions(
        jsonl_path: str,
        cls_id: int,
        image_ids: List[str],
        captions_texts: List[str],
        class_name: Optional[str] = None,
) -> None:
    """
    结构：{"class": int, "image_ids": [..K..], "captions": [{"text": ...}, ...]}
    """
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    ids = [str(s).replace("\\", "/") for s in image_ids]
    cap_texts = [{"text": str(t).strip()} for t in captions_texts]

    rec = {
        "class": int(cls_id),
        "class_name": class_name or str(cls_id),
        "image_ids": ids,
        "captions": cap_texts,
        # "time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[caption-cache] wrote class {cls_id} : total {len(cap_texts)} captions to: {jsonl_path}")


# 1028 add
def labels_to_text(ids, dataset: str):
    """
    把类别索引转成文本，若输入 list 则返回全部文本，否则返回指定 id 文本。
    """
    fname = f"data/{dataset}/class_names.txt"
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    names = [" ".join(line.split()[1:]) for line in lines]  # 允许多单词的类名

    return [names[i] for i in ids]


# 1028 add  计算类的个数
def count_labels_nums():
    path = "data/coco/class_names.txt"
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# 1028 add
def get_class_bank(net, device, config, cache_path):
    # 如果缓存目录有文件直接载入并返回
    if os.path.isfile(cache_path):
        class_bank = torch.load(cache_path, map_location=device)
        print(f"[class_bank] load cache at: {cache_path}")
        return class_bank if class_bank.device == device else class_bank.to(device)

    # 否则用text_encoder重新生成并保存为class_bank.pt
    num_classes = count_labels_nums()  # 类别数
    class_texts = labels_to_text(list(range(num_classes)),
                                 dataset=config["dataset"])  # 获取class names ['person', 'bicycle', ...]
    print(f"[class_bank] classes number: {num_classes}  (first 5: {class_texts[:5]})")
    t0 = time.time()
    with torch.no_grad():
        class_bank = net.text_encoder.encode(class_texts).to(device)  # [C,256]
        print(f"[class_bank] encode time: {time.time() - t0:.3f}s, shape={tuple(class_bank.shape)}")
    # 缓存到 CPU 文件，训练时再 to(device)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(class_bank.detach().cpu(), cache_path)
    print(f"[class_bank] save class bank at: {cache_path}")
    return class_bank


def make_prompt_texts(prompt: str, class_names: List[str]) -> List[str]:
    """
    针对 class-level prompt（如“a photo of cat”）的通用构造函数
    根据 prompt 模板和类名生成文本描述。
    兼容三种情况：
      - prompt 为空：直接返回类名；
      - prompt 含 "{}" 或 "{name}"：按模板格式化；
      - prompt 无占位符：前缀拼接。
    示例：
        make_prompt_texts("", ["cat","dog"])
            → ["cat","dog"]
        make_prompt_texts("a photo of", ["cat","dog"])
            → ["a photo of cat","a photo of dog"]
    """
    if not prompt or not prompt.strip():
        return class_names
    p = prompt.strip()
    if "{name}" in p:
        return [p.format(name=n) for n in class_names]
    if "{}" in p:
        return [p.format(n) for n in class_names]
    return [f"{p} {n}".strip() for n in class_names]


# add 1112
def make_prompt_for_captions(prompt: str, texts):
    """
    对句子列表或嵌套的词列表加 prompt，如果没有直接返回（and拼接后返回），有则将prompt作为profix返回
        ① List[str]      → ["a photo of cat", "a photo of dog"]
        ② List[List[str]]→ ["a photo of cat and dog", "a photo of car and road"]
        1. 输入为嵌套关键词列表：
            prompt = "a photo of"
            texts = [["cat", "dog"], ["car", "road"]]
            输出：
                ["a photo of cat and dog", "a photo of car and road"]
        2. 输入为句子列表：
            prompt = "a photo of"
            texts = ["a small cat", "a running dog"]
            输出：
                ["a photo of a small cat", "a photo of a running dog"]
    """
    if not prompt or not prompt.strip():
        if all(isinstance(x, list) for x in texts):
            # 嵌套关键词列表
            return [" and ".join(inner) for inner in texts]
        else:
            # 普通句子列表，直接返回
            return list(texts)
    else:
        p = prompt.strip()

        if all(isinstance(x, list) for x in texts):
            # 嵌套关键词列表：内部用 and 连接
            return [f"{p} {' and '.join(inner)}".strip() for inner in texts]
        else:
            # 普通句子列表：直接加前缀
            return [f"{p} {str(t).strip()}".strip() for t in texts]


# add 1105
def extract_keywords(
        text: str,
        kw_model: KeyBERT,
        top_n: int = 5,
        min_ngram: int = 1,
        max_ngram: int = 2
) -> List[Tuple[str, float]]:
    """
    功能：对单条文本用 KeyBERT 抽取关键词短语
    参数：
        text: 原始 caption 文本
        kw_model: KeyBERT 实例
        top_n: 返回的候选数
        min_ngram/max_ngram: 关键词长度范围（词数）
    返回：
        列表[(关键词, 分数)]
    说明：
        - use_mmr=True 增强多样性；diversity 可按需调节。
    """
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(min_ngram, max_ngram),
        stop_words="english",
        use_maxsum=False,
        use_mmr=True,
        diversity=0.5,
        top_n=top_n
    )


# add 1105
def pos_filter_terms(terms: List[str], nlp) -> List[str]:
    """
    功能：对候选词做词性过滤，仅保留名词/专有名词/形容词。
    参数：
        terms: 关键词列表（不含分数）
        nlp: spaCy English pipeline
    返回：
        过滤后的关键词列表
    """
    kept = []
    for t in terms:
        doc = nlp(t)
        # 对短语取所有 token 的 POS，只要包含 NOUN/PROPN/ADJ 即保留
        if any(tok.pos_ in {"NOUN"} for tok in doc):
            # if any(tok.pos_ in {"NOUN", "PROPN", "ADJ"} for tok in doc):
            kept.append(t)
    # 去重并保序
    seen = set()
    uniq = []
    for x in kept:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# 1105 add
def load_filtered_captions_jsonl(jsonl_path: str):
    """
    从 filtered_captions.jsonl 读取结果，
    返回结构：
        { class_name: {"captions": [...], "filtered_captions": [[...], ...]} }
    """
    out = {}
    if not os.path.isfile(jsonl_path):  # 没有则读到空返回
        return out

    # 正常读
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.strip())
                cid = obj.get("class_id")
                cls_name = str(obj.get("class_name", obj.get("class", "")))
                caps = obj.get("captions")
                filtered_caps = obj.get("filtered_captions", [])
                out[cid] = {
                    "class_name": cls_name,
                    "captions": caps,
                    "filtered_captions": filtered_caps,
                }
            except Exception:
                pass
    return out


# 1114 add
def get_filtered_captions(
        config,
        top_n: int = 4,
        min_ngram: int = 1,
        max_ngram: int = 2,
        score_threshold: Optional[float] = None,
):
    """
    功能：返回“过滤后的 captions（逐条一一对应的关键词列表）”，并按需写入/复用缓存：
        - 若 filtered_jsonl_path 已存在：直接读取并返回 {class_name: List[List[str]]}
        - 否则：使用既有 load_class_captions(caption_cache_path) 读取每类多条 caption，
                对每条 caption 执行 KeyBERT(+可选 POS/+可选阈值)，
                一类一行写入 filtered_jsonl_path：
                {"class": name, "captions": [...], "filtered_captions": [[...], ...]}
    参数：
        caption_cache_path : 旧版 captions.jsonl（请使用已实现的 load_class_captions 读取）
        filtered_jsonl_path: 目标输出 JSONL 路径（新版结构）
        model_dir          : KeyBERT 底座（Sentence-Transformers 本地目录）
        top_n/min_ngram/max_ngram/score_threshold: KeyBERT 控制项与分数阈值
    返回：
        {class_name: List[List[str]]}，与“每类 captions 的长度”一一对应
    """
    filtered_jsonl_path = config.get("filtered_caps_path")
    caption_cache_path = config.get("caption_save_path")
    dataset = config.get("dataset")
    kb_dir = config.get("keybert_model_dir")

    # A) 若过滤结果已存在，则读取
    if os.path.isfile(filtered_jsonl_path):
        cached = load_filtered_captions_jsonl(filtered_jsonl_path)
        if cached:
            print(f"[filtered-captions] load cache at: {filtered_jsonl_path} ({len(cached)} classes)")
            return {c: cached[c]["filtered_captions"] for c in cached}

    # B) 无缓存则生成：读取“每类多条 caption”
    class2caps = load_class_captions(caption_cache_path)  # 外部已实现
    class_ids = class2caps.keys()

    if not class_ids:
        print(f"[filtered-captions] WARN: no source captions found at: {caption_cache_path}")

    # C) 模型：KeyBERT + spaCy
    kw_model = KeyBERT(model=kb_dir)
    nlp = spacy.load("en_core_web_sm")  # spacy英文语言模型，不支持中文处理，中文请移步：spacy-zh-core-web-sm等

    # D) 处理逻辑
    results = {}
    os.makedirs(os.path.dirname(filtered_jsonl_path) or ".", exist_ok=True)

    with open(filtered_jsonl_path, "w", encoding="utf-8") as wf:

        for c in class_ids:
            caps = [str(t).strip() for t in (class2caps.get(c) or []) if str(t).strip()]
            if not caps:
                caps = [f"class {c}"]

            cls_name = labels_to_text([int(c)], dataset=dataset)[0].strip().lower()
            per_caption_terms: List[List[str]] = []

            for cap in caps:
                # 1) KeyBERT 抽取候选 phrases
                pairs = extract_keywords(
                    cap, kw_model,
                    top_n=top_n, min_ngram=min_ngram, max_ngram=max_ngram
                )
                # 2) 分数过滤
                terms = [
                    w.strip().lower()
                    for (w, s) in pairs
                    if (score_threshold is None or float(s) >= float(score_threshold))
                ]
                # 3 POS 过滤，仅保留名词/专有名词, 这一步摒弃了之前的 pos_filter_terms()
                kept = []
                for t in terms:
                    doc = nlp(t)
                    if any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc):
                        kept.append(t)
                # 4) class_name 放在第一位 + 去重
                seen = set()
                uniq_terms = []
                # (a) 先放 class_name
                uniq_terms.append(cls_name)
                seen.add(cls_name)
                # (b) 再放 kept 里的 terms（原顺序 + 去重）
                for t in kept:
                    if t and t not in seen:
                        uniq_terms.append(t)
                        seen.add(t)
                per_caption_terms.append(uniq_terms)
            # 写入 JSONL（一类一行）
            rec = {
                "class_id": c,
                "class_name": cls_name,
                "captions": caps,
                "filtered_captions": per_caption_terms
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results[c] = per_caption_terms

        print(f"[filtered-captions] wrote {len(results)} classes to: {filtered_jsonl_path}")
        # 清理 KeyBERT / spaCy 内存
        try:
            del kw_model
            del nlp
        except:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def clean_save_dir_keep_best(save_dir, dataset_name):
    """
    清理 save 目录中除最佳 mAP 文件外的所有模型/中间文件
    """

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
    for img, cls, _, _ in tqdm(dataloader):
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


# =========================
# 训练前一次性预生成类级 captions
# =========================
@torch.no_grad()
def precompute_class_captions(
        captioner,
        config,
        device: torch.device = torch.device("cuda")
):
    """
    功能：
        - 每类采样 captions_num 张图片（K）
        - 每张图生成 1 条 caption（top-1）
        - 以 JSONL 写入：{"class", "image_ids", "captions":[{"text":...}, ...]}
        - 若缓存已存在且非空，则直接读取
    返回：
        class2caps: {class_id: [caption1, caption2, ..., captionK]}
    """
    captions_num = int(config.get("captions_num", 3))
    caps_cache_path = config.get("caption_save_path")
    dataset = config.get("dataset")

    # 如果文件存在且非空，跳过生成并直接读
    if os.path.isfile(caps_cache_path) and os.path.getsize(caps_cache_path) > 0:
        print(f"[caption-cache] found existing captions → skip generation")
        return load_class_captions(caps_cache_path)
    else:
        print(f"[caption-cache] not found or empty → generating new captions...")

        # 构建 caption 专用 DataLoader（使用 train_all.txt）
        caption_loader = build_caption_loader(config)

        num_classes = count_labels_nums()  # 类别数
        K = captions_num  # 每类采样 K 张图
        class2caps: Dict[int, List[str]] = {}

        # 每类采样容器
        """
        picked_imgs = {
        0: [img_tensor0_1, img_tensor0_2, img_tensor0_3],   # airplane 类的3张图
        1: [img_tensor1_1, img_tensor1_2, img_tensor1_3],   # automobile 类的3张图
        ...}
        picked_abs = {
        0: [".../airplane/001.png", ".../airplane/002.png", ".../airplane/003.png"],
        1: [".../automobile/101.png", "..."],
        ...}
        先遍历整个 train_loader；
        把属于每个类的图像（张量 + 路径）放进各自的 list；    
        当数量达到 captions_num（比如 3）就停止往该类里放；    
        所有类都达到 captions_num 后跳出循环；    
        最后再对每个类的 list 统一送入 captioner.generate()。
        """
        picked_imgs = {c: [] for c in range(num_classes)}  # list[Tensor[C,H,W]] up to K 存放每个类别采样到的图像 tensor
        picked_paths = {c: [] for c in range(num_classes)}  # 对应磁盘路径（用于写入 image_ids） 存放每个类别采样到的 图像路径
        class_texts = labels_to_text(list(range(num_classes)), dataset=dataset)  # ["person","bicycle",...]

        # 遍历 train_loader，为每类收集 K 张图
        for images, labels, ind, paths in caption_loader:
            # labels: [B, C] → 逐行找到所有 >0 的 index（适用于 one-hot/multi-hot）
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            labels = labels.long()

            for b in range(labels.size(0)):
                cls_list = torch.nonzero(labels[b] > 0).view(-1).tolist()

                for c in cls_list:
                    if len(picked_imgs[c]) < K:
                        picked_imgs[c].append(images[b].cpu())
                        picked_paths[c].append(paths[b])
            # 若全部类都收齐 K 张图 → 提前退出
            if all(len(picked_imgs[c]) >= K for c in range(num_classes)):
                break

        t0 = time.time()
        # 为每类批量生成 captions（无 prompt）
        for c in range(num_classes):
            imgs = torch.stack(picked_imgs[c]).to(device)  # [k,3,H,W]
            texts, scores = captioner.generate(imgs, return_scores=True)
            cap_texts = [t.strip() for t in texts]
            # 落盘：image_ids 使用原始路径（已是绝对路径），统一为正斜杠
            image_ids = [p.replace("\\", "/") for p in picked_paths[c]]
            append_class_captions(
                jsonl_path=caps_cache_path,
                cls_id=int(c),
                image_ids=image_ids,
                captions_texts=cap_texts,  # 接口改名后的参数
                class_name=class_texts[c],
            )

            class2caps[c] = cap_texts if cap_texts else [f"class {c}"]
        cost = time.time() - t0
        print(f"[caption-precompute] class {c}: k={len(cap_texts)}, time={cost:.2f}s")

    return class2caps


# ====== t-SNE ===================================================
# 从 dataloader 中提取用于可视化的一批连续特征（不取 sign），同时抽出标签做着色。

def collect_features_for_tsne(dataloader,
                              net,
                              device,
                              max_points: int = 4000):
    """
    从 dataloader 顺序收集样本，提取“连续特征”并返回（用于 t-SNE）。
    约定：
      - dataloader 的每个 batch 返回：(img, cls, _, _) 结构；
      - net(img) 返回未二值化的连续向量（若你的 forward 返回线性输出，t-SNE 更稳；
        我们这里额外做了一次 tanh，把范围规约到 (-1,1) 以抑制极端值）。
    参数：
      dataloader : torch.utils.data.DataLoader
      net        : 已加载权重、处于 eval() 或 train() 任意状态的模型
      device     : torch.device
      max_points : 最多采多少个点来可视化（越大会越慢/越吃显存/内存）
    返回：
      features_np: (N, D) 的 numpy.ndarray  连续特征
              N：最终采集到的样本数（不超过 max_points）；
              D：模型输出的特征维度
      labels_np  : (N,)   的 numpy.ndarray  每个样本的可视化标签（单标签的类别 id）
    """
    net.eval()
    feats, labs = [], []
    collected = 0

    with torch.no_grad():
        for img, cls, _, _ in dataloader:  # 与你的 ImageList __getitem__ 对齐
            img = img.to(device, non_blocking=True)

            # 1) 前向拿到连续特征（不做 sign）
            #    你的 net.forward 返回的是哈希头线性输出；我们做 tanh 便于可视化的稳定性
            u = net(img).tanh().detach().cpu()  # [B, bit]

            # 2) 标签处理：如果是 one-hot / multi-hot，用 argmax 做一个可视化分组
            if isinstance(cls, np.ndarray):
                cls = torch.from_numpy(cls)  # 把class转换为张量tensor
            cls_cpu = cls.detach().cpu()
            if cls_cpu.ndim == 1:
                lab = cls_cpu.long()  # 单标签，直接用
            else:  # 用“随机挑一个正类”替代 argmax（且把全 0 行标成 -1 用灰色显示）
                picked = []
                for row in cls_cpu:  # row: [n_class]
                    pos = torch.nonzero(row > 0).view(-1)
                    if len(pos) == 0:
                        picked.append(torch.tensor(-1))  # 全 0 → 无标签，用 -1 标记
                    else:
                        j = torch.randint(0, len(pos), (1,)).item()
                        picked.append(torch.tensor(int(pos[j])))
                lab = torch.stack(picked, dim=0).long()

            feats.append(u)
            labs.append(lab.cpu())
            collected += u.size(0)

            if collected >= max_points:
                break

    features_np = torch.cat(feats, dim=0)[:max_points].numpy()
    labels_np = torch.cat(labs, dim=0)[:max_points].numpy()
    return features_np, labels_np


# - tsne_plot：把高维特征降到 2D 并保存散点图。
def tsne_plot(features,
              labels=None,
              save_path: str = "tsne.png",
              seed: int = 42,
              perplexity: float = 30.0,
              learning_rate: float = 200.0,
              n_iter: int = 1000):
    """
    用 t-SNE 把高维特征降到二维，并保存散点图。
    参数：
      features     : numpy.ndarray, shape=(N, D)，高维输入特征
      labels       : numpy.ndarray 或 None, shape=(N,)，用于着色的类别 id（可选）
      save_path    : 输出图片路径（.png）
      seed         : 随机种子（保证复现）
      perplexity   : t-SNE 困惑度；类簇较多/数据更多时可以适当增大（但通常 5~50）
      learning_rate: 学习率；默认 200，过小收敛慢，过大可能不稳定
      n_iter       : 迭代步数；1000 对小中等规模通常足够
    返回：
      coords       : numpy.ndarray, shape=(N, 2)，二维坐标
    """
    matplotlib.use("Agg")  # 改成纯文件输出后端
    # 1) 运行 t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate=float(learning_rate),
        max_iter=int(n_iter),
        init="pca",
        random_state=int(seed),
        verbose=0,
        metric="euclidean"
    )
    coords = tsne.fit_transform(features)  # (N, 2)

    # 2) 画散点
    plt.figure(figsize=(6, 5), dpi=150)
    if labels is None:
        plt.scatter(coords[:, 0], coords[:, 1], s=4, alpha=0.7)
        plt.title("t-SNE (no labels)")
    else:
        # 为了简单稳妥，直接用 matplotlib 的默认颜色循环；类别多也能自动回退
        num_classes = int(np.max(labels)) + 1
        for c in range(num_classes):
            mask = (labels == c)
            if not np.any(mask):
                continue
            plt.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.75, label=str(c))
        # 类别太多时图例可能太挤，可按需注释掉下一行
        if num_classes <= 20:
            plt.legend(markerscale=2, frameon=True)

        plt.title("t-SNE by class")

    plt.xticks([])
    plt.yticks([])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[t-SNE] saved to: {save_path}")
    return coords
