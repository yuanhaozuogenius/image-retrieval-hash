import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.optimizeAccel import build_hash_centers
from utils.tools import count_labels_nums
import os

class CSQPlusLoss(nn.Module):
    def __init__(self, config, bit):
        super().__init__()
        self.config = config
        self.bit = bit

        dataset = config.get("dataset")
        n_class = count_labels_nums()
        center_path = f"./data/{dataset}/CSQ_init_True_{n_class}_{bit}.npy"

        if not os.path.exists(center_path):
            print(
                f"[CSQPlusLoss] Hash centers not found → Generating for dataset '{dataset}' ({n_class} classes, {bit}-bit)")
            hash_centers = build_hash_centers(
                n_class=n_class,
                bit=bit,
                dataset=dataset,
                initWithCSQ=True,
                save=True
            )
        else:
            print(f"[CSQPlusLoss] Using existing centers: {center_path}")
            hash_centers = np.load(center_path)

        # === 缓存路径并转换为 Tensor === #
        self.hash_center = torch.from_numpy(hash_centers).float().cuda()
        self.label_center = torch.eye(config['n_class']).float().cuda()  # one-hot 类别向量

    def forward(self, u, y):
        """
        u: [batch, bit] 经过ResNet + tanh输出的连续哈希向量
        y: [batch, n_class] one-hot标签向量
        """
        # -------------------------
        # 1. L_C: center similarity loss
        # -------------------------
        u_norm = F.normalize(u, p=2, dim=1)
        centers_norm = F.normalize(self.hash_center, p=2, dim=1)
        cos_sim = torch.matmul(u_norm, centers_norm.t())  # [batch, n_class]
        cos_sim = (self.bit ** 0.5) * cos_sim  # scaled cosine
        log_prob = F.log_softmax(cos_sim, dim=1)
        L_C = -(y * log_prob).sum(dim=1).mean()  # cross-entropy style

        # -------------------------
        # 2. L_P: pairwise similarity loss (同类样本内积越大越好)
        # -------------------------
        label_sim = torch.matmul(y, y.t())  # [batch, batch], 1表示同类
        u_inner = torch.matmul(u, u.t())  # [batch, batch]
        L_P_matrix = torch.log1p(torch.exp((self.bit - u_inner) / (2 * self.bit)))
        # 只考虑同类对
        mask = (label_sim > 0).float()
        if mask.sum() > 0:
            L_P = (L_P_matrix * mask).sum() / mask.sum()
        else:
            L_P = torch.tensor(0.0).cuda()

        # -------------------------
        # 3. L_Q: quantization loss
        # -------------------------
        # L_Q = ((u.abs() - 1.0)**2).mean()
        L_Q = (u.abs() - 1.0).abs().mean()

        # -------------------------
        # 总损失
        # -------------------------
        loss = L_C + self.config['beta'] * L_P + self.config['lambda'] * L_Q

        return loss, L_C, L_P, L_Q
