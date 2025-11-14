import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 可选：将 BLIP 子目录添加用于导入模型
blip_path = os.path.join(ROOT, 'BLIP')
blip2_path = os.path.join(ROOT, 'BLIP2')
if blip_path not in sys.path or blip2_path not in sys.path:
    sys.path.append(blip_path)
    sys.path.append(blip2_path)

from utils.tools import *

from BLIP.models.blip_itm import blip_itm
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import os, json, random, pickle
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import time
from BLIP2.to_vector import build_text_encoder
from BLIP2.to_label import build_image_captioner
import gc

torch.multiprocessing.set_sharing_strategy('file_system')


def build_blip_net(bit, **kwargs):
    return BLIP_HashWrapper(bit, **kwargs)


def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ_BLIP_3]",
        "resize_size": 224,
        "crop_size": 224,
        "batch_size": 64,
        "net": build_blip_net,
        "dataset": "coco",
        "epoch": 120,
        "eval_switch_epoch": 60,  # 前60轮 → test_map_1；后60轮 → test_map_2
        "test_map_1": 30,  # 前半段评估间隔
        "test_map_2": 10,  # 后半段评估间隔
        "device": torch.device("cuda:0"),
        "bit_list": [64],
        "save_path": "save/CSQ_BLIP_3",
        # "cifar10_dir": r"D:\Datasets\cifar10-image",  # 后续将实际数据路径设置在 image_root 中 方便统一修改
        "image_root": r"D:\Datasets\coco2017",
        # —— 跨模态对齐控制 —— #
        # 对齐损失权重（0.5~2.0 之间微调）
        # 若发现 CSQ 很好但对齐很弱（Align 维持很小且总 mAP 没涨），试着加到 1.5–2.0；
        # 若 mAP 掉得厉害，往往对齐过强或干扰量化，试着降到 0.3–0.7。
        "align_weight": 1.0,
        "align_mode": "mse",  # "mse" 或 "cosine"
        "text_proj_dim": 256,  # 文本侧投影维度（与视觉侧 256 对齐）
        "text_anchor_weight": 0.05,  # 文本侧恒等锚定正则的权重 β（防止 fc2 漂移过头）建议 0.01~0.1
        "text_adapter_lr_mult": 0.5,  # 给文本adapter一个更小的 lr（乘数）

        # 图生文
        "caption_num_beams": 1,  # 建议 ≥2 以保证 sequences_scores 可用
        "caption_max_new_tokens": 32,  # 句长上限（越大越慢）
        "captions_num": 10,  # 固定采样数 K
        "caption_prompt": "a photo of a",  # 提示词（caption 前缀） cifar不建议加，且BLIP-2对CIFAR这种小图的视觉辨识力有限

        # —— 相对路径—— #
        "blip_dir": r"D:\Models\blip2-opt-2.7b",
        "all_train_data": r"./data/{dataset}/train_all.txt",
        "caption_save_path": "./data/{dataset}/captions.jsonl",  # 保存图像生成的caption文本，
        "fc_path": "./trained_mappers/image_mapper.pth",  # 若没有可注释掉加载
        "med_config": "BLIP/configs/med_config.json",
        "blip_ckpt": "./models/model_base.pth"
    }
    config = config_dataset(config)

    # 将 {dataset} 替换为真实数据集名
    config["caption_save_path"] = config.get("caption_save_path").replace("{dataset}", config["dataset"])
    config["all_train_data"] = config.get("all_train_data").replace("{dataset}", config["dataset"])
    return config


#  —— CIFAR-10 类名
CIFAR10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


class FeatureMapper(nn.Module):
    def __init__(self, input_dim=256, output_dim=64):
        super(FeatureMapper, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输出给CSQ，数值范围不强约束，这里只做线性映射；归一化在损失/对齐时再做
        return self.fc(x)

    """
    - BLIP 主体冻结，仅作为视觉编码器（产生 v256）
    - img_adapter: 256→256 可训练；在它的输出与文本 t256 上做对齐损失
    - mapper: 256→bit 可训练；最终哈希向量 u = mapper(img_adapter(v256))
    - forward(image) 仅返回 u（bit），保持 compute_result 兼容
    - encode_image_to_256(image) 返回冻结的 v256，供训练循环单独拿
    """


class BLIP_HashWrapper(nn.Module):
    def __init__(self, bit,
                 blip_ckpt='./models/model_base.pth',
                 med_config='BLIP/configs/med_config.json',
                 fc_path='./trained_mappers/image_mapper.pth',
                 text_proj_dim=256, blip_dir=r"D:\Models\blip2-opt-2.7b"):
        super().__init__()
        # —— BLIP 视觉侧（冻结）—— #
        self.model = blip_itm(pretrained=blip_ckpt, med_config=med_config, image_size=224, vit='base')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ——fc1： 图像侧适配器（可训练，256→256）：对齐发生在它的输出上 —— #
        # 简单稳妥的小MLP：Linear + LayerNorm + GELU + Linear
        self.img_adapter = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256)
        )

        # ——fc2: 文本侧适配头（可训练，256→256） —— #
        self.text_adapter = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256)
        )

        # —— 哈希映射（可训练，256→bit）—— #
        self.mapper = FeatureMapper(256, bit)
        if fc_path and os.path.exists(fc_path):
            self.mapper.load_state_dict(
                torch.load(fc_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        # 文本编码器（你在 to_vector.py 里已完成）
        self.text_encoder = build_text_encoder(model_dir=blip_dir, proj_dim=text_proj_dim,
                                               device="cuda" if torch.cuda.is_available() else "cpu")
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    # v256 = “要与文本对齐的语义空间特征向量 feature”（fc1），↔ 文本 256 维
    @torch.no_grad()
    def encode_image_to_256(self, image):
        # 冻结 BLIP，取 CLS 再做 L2 归一化
        vision_embeds = self.model.visual_encoder(image)  # 经过Transformer(viT)编码
        v256 = self.model.vision_proj(vision_embeds[:, 0, :])  # 第0个位置——>CLS 代表该图的特征 [B, 256]
        v256 = F.normalize(v256, dim=-1)
        return v256

    def forward(self, image):
        # 注意：只返回 u（bit）
        with torch.no_grad():
            v256 = self.encode_image_to_256(image)  # [B,256] 冻结 BLIP
        v_adapt = self.img_adapter(v256)  # [B,256] 可训练
        u = self.mapper(v_adapt)  # [B,bit]  可训练
        # u = 要用于哈希检索的压缩向量
        return u


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])
        self.last_center_loss = 0.0
        self.last_q_loss = 0.0

    def forward(self, u, y, ind, config):
        # u 连续 -> tanh 压到 (-1,1)，与中心的二值(-1/1)做 BCE（映射到[0,1]）
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        Q_loss = (u.abs() - 1).pow(2).mean()
        self.last_center_loss = center_loss.item()
        self.last_q_loss = Q_loss.item()
        return center_loss + config["lambda"] * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = [sum(hash_targets[i] != hash_targets[j]) for i in range(n_class) for j in range(i)]
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets


# —— 对齐损失（发生在 img_adapter(v256) ↔ t_adapt）—— #
class AlignmentLoss(nn.Module):
    def __init__(self, mode="mse"):
        super().__init__()
        self.mode = mode
        self.mse = nn.MSELoss()

    # forward写法比较方便后续实例化和取，v_adapt = img_adapter(v256)
    def forward(self, v_adapt, t_adapt):
        v_adapt = F.normalize(v_adapt, dim=-1)
        t_adapt = F.normalize(t_adapt, dim=-1)
        if self.mode == "mse":
            return self.mse(v_adapt, t_adapt)
        elif self.mode == "cosine":
            cos = F.cosine_similarity(v_adapt, t_adapt, dim=-1)
            return (1.0 - cos).mean()
        else:
            raise ValueError("Unsupported align mode")


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    n_class = config.get("n_class") or count_labels_nums()

    # 图生文 captioner（仅推理）
    captioner = build_image_captioner(
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_dir=config.get("blip_dir", r"D:\Models\blip2-opt-2.7b"),
        num_beams=int(config.get("caption_num_beams", 1)),
        max_new_tokens=int(config.get("caption_max_new_tokens", 32))
    )

    # ———— Caption 缓存路径与初始化 ———— #
    # ———— 训练前一次性预生成类级 captions ———— #
    class2caps = precompute_class_captions(
        captioner=captioner,
        config=config,
        device=device
    )
    prompt = config.get("caption_prompt", "")
    # 容器：用于保存所有类的 prompt 后 captions
    class2caps_prompt = {}
    for c in range(n_class):
        captions = class2caps.get(c, None)  # 取该类原始 caption 列表（list[str]）
        # if not captions: #
        #     continue
        prompted_caps = make_prompt_for_captions(prompt, captions)
        class2caps_prompt[c] = prompted_caps

    # 释放 captioner 显存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del captioner
    gc.collect()  # 清理已经没有引用的对象
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 清空缓存后再实例化net
    net = config["net"](bit,
                        blip_ckpt=config.get("blip_ckpt", "./models/model_base.pth"),
                        med_config=config.get("med_config", "BLIP/configs/med_config.json"),
                        fc_path=config.get("fc_path", "./trained_mappers/image_mapper.pth"),
                        text_proj_dim=config.get("text_proj_dim", 256),
                        blip_dir=config.get("blip_dir", config.get("model_dir", r"D:\Models\blip2-opt-2.7b")),
                        ).to(device)

    # print(f"[caption-cache] loaded {len(class2cap)} classes from {caps_cache_path}")

    # 3) Caption bank（样本全集）
    texts = []

    # 将“每条 caption 的关键词列表”拼成一句短语（"; " 连接），逐条写入负样本库
    for c in range(n_class):
        term_lists = class2caps_prompt.get(c, None)
        if term_lists and len(term_lists) > 0:
            for text in term_lists:
                if text:
                    texts.append(text)

    with torch.no_grad():
        # caption_bank : captions转换的张量。用于构建t256
        caption_bank = net.text_encoder.encode(texts).to(device)  # [Nneg, D]
        caption_bank = F.normalize(caption_bank, dim=-1)

    # 优化器（分组：fc1 / hash head / fc2）
    base_optim = config["optimizer"]
    base_lr = base_optim["optim_params"].get("lr", 1e-5)
    ta_mult = config.get("text_adapter_lr_mult", 0.5)
    # 只找params中的最优解
    params = [
        {"params": net.img_adapter.parameters(), "lr": base_lr},  # fc1
        {"params": net.mapper.parameters(), "lr": base_lr},  # hash head
        {"params": net.text_adapter.parameters(), "lr": base_lr * ta_mult},  # ★ fc2 稍低 lr 更稳
    ]
    optimizer = base_optim["type"](params, **{k: v for k, v in base_optim["optim_params"].items() if k != "lr"})

    # 初始化loss对象
    csq_criterion = CSQLoss(config, bit)
    align_criterion = AlignmentLoss(mode=config.get("align_mode", "mse"))
    align_weight = float(config.get("align_weight", 1.0))
    beta = float(config.get("text_anchor_weight", 0.05))
    # 扫描当前 save/算法名/ 文件夹，找到得分最高（mAP 最大）的一组前缀，然后删除其余所有 .pt 和 .npy 文件，只保留那一组
    # if "save_path" in config:
    #     clean_save_dir_keep_best(config["save_path"], config["dataset"])
    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        # —— 动态评估间隔 —— #
        if epoch < config["eval_switch_epoch"]:
            eval_interval = config["test_map_1"]
        else:
            eval_interval = config["test_map_2"]

        net.train()
        train_loss = 0.0
        csq_loss_meter = 0.0
        center_loss_meter = 0.0
        q_loss_meter = 0.0
        align_loss_meter = 0.0
        id_loss_meter = 0.0

        for images, labels, ind, paths in train_loader:

            # ====== 进入一个 batch ======
            # —— 搬到 GPU —— #
            images = images.to(device, non_blocking=True)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            labels = labels.to(device).float()
            if torch.cuda.is_available(): torch.cuda.synchronize()

            # —— 文本侧 —— #
            with torch.no_grad():
                if labels.ndim == 2 and labels.size(1) == 1:  # 多标签
                    y_idx = labels.view(-1).long()
                else:  # 非多标签（单标签）
                    y_idx = labels.argmax(dim=1).long()

                # 调试更稳：用 CPU 做索引再搬回 GPU，避免 GPU 高级索引隐式同步
                y_idx_cpu = y_idx.detach().cpu()
                t256 = caption_bank[y_idx_cpu].to(device)  # 从caption_bank取出y_index对应批的tensor

            if torch.cuda.is_available(): torch.cuda.synchronize()

            # —— 清梯度，防止梯度累计—— #
            optimizer.zero_grad(set_to_none=True)

            # —— 视觉侧（冻结 BLIP 编码到 256） —— #
            with torch.no_grad():
                v256 = net.encode_image_to_256(images)  # [B,256]
            if torch.cuda.is_available(): torch.cuda.synchronize()

            # —— 适配层与哈希头 —— #

            v_adapt = net.img_adapter(v256)  # [B,256]
            t_adapt = net.text_adapter(t256)  # [B,256]
            if torch.cuda.is_available(): torch.cuda.synchronize()

            u = net.mapper(v_adapt)  # [B,bit]
            if torch.cuda.is_available(): torch.cuda.synchronize()

            # —— 损失 —— #
            # fc1 fc2 的结果做对齐，而不是fc1 fc2网络结构对齐
            align_loss = align_criterion(v_adapt, t_adapt)
            # 让t_adapt和 t256的乘积最小——>最相似
            id_loss = F.mse_loss(F.normalize(t_adapt, dim=-1), t256)
            csq_loss = csq_criterion(u, labels, ind, config)

            if torch.cuda.is_available(): torch.cuda.synchronize()

            total_loss = csq_loss + align_weight * align_loss + beta * id_loss

            total_loss.backward()

            # 优化器：获取最优解
            optimizer.step()
            if torch.cuda.is_available(): torch.cuda.synchronize()

            # —— 统计 —— #
            train_loss += total_loss.item()
            csq_loss_meter += csq_loss.item()
            center_loss_meter += csq_criterion.last_center_loss
            q_loss_meter += csq_criterion.last_q_loss
            align_loss_meter += align_loss.item()
            id_loss_meter += id_loss.item()

        # 统计并打印 每个 batch 的 loss 会抖动,把整轮（epoch）里所有 batch 的 loss 求平均
        n_iter = len(train_loader)  # batch 数
        train_loss_avg = train_loss / n_iter
        csq_loss_avg = csq_loss_meter / n_iter
        center_loss_avg = center_loss_meter / n_iter
        q_loss_avg = q_loss_meter / n_iter
        align_loss_avg = align_loss_meter / n_iter
        id_loss_avg = id_loss_meter / n_iter

        print(
            f"{config['info']}[{epoch + 1:>2}/{config['epoch']}][{current_time}] "
            f"bit:{bit}, dataset:{config['dataset']}, "
            f"total loss:{train_loss_avg:.3f} = "
            f"CSQ:{csq_loss_avg:.3f} (L_C:{center_loss_avg:.3f} + L_Q:{q_loss_avg:.3f}) "
            f"+ Align:{align_loss_avg:.3f} + ID:{id_loss_avg:.3f} "
            f"(α={align_weight:.2f}, β={beta:.2f}, mode:{config.get('align_mode', 'mse')})"
        )
        # 评估与保存
        if (epoch + 1) % eval_interval == 0:
            tst_binary, tst_label = compute_result(test_loader, net, device=device)
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            if mAP > Best_mAP:
                Best_mAP = mAP
                if "save_path" in config:
                    # if not os.path.exists(config["save_path"]):
                    #     os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    save_path = config["save_path"]
                    # 替换 "-" → "_"，确保正则或文件名不会误解析
                    dataset_tag = config["dataset"].replace("-", "_")
                    # 格式化 MAP 保留固定小数位，避免路径长度混乱
                    score_str = f"{mAP:.10f}"
                    filename_prefix = f"{dataset_tag}-{score_str}"
                    # 保存模型及中间文件
                    # np.save(os.path.join(save_path, f"{filename_prefix}-trn_binary.npy"), trn_binary.numpy())
                    # np.save(os.path.join(save_path, f"{filename_prefix}-tst_binary.npy"), tst_binary.numpy())
                    # np.save(os.path.join(save_path, f"{filename_prefix}-trn_label.npy"), trn_label.numpy())
                    # np.save(os.path.join(save_path, f"{filename_prefix}-tst_label.npy"), tst_label.numpy())
                    # torch.save(net.state_dict(), os.path.join(save_path, f"{filename_prefix}-model.pt"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            # print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
    print(">>> script reached end of file")
