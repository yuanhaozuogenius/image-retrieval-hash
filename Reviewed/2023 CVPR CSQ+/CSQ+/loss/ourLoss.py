from math import gamma
import torch
import torch.nn as NN
from scipy.linalg import hadamard, eig
import numpy as np
import random
# from numpy import *
from scipy.special import comb
from loguru import logger
import pdb
import itertools
import copy
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

class OurLoss(nn.Module):
    def __init__(self, config, bit, l):
        super(OurLoss, self).__init__()
        self.config = config
        self.bit = bit
        self.alpha_pos, self.alpha_neg, self.beta_neg, self.d_min, self.d_max = self.get_margin()
        self.hash_center = self.generate_center(bit, config['n_class'], l)
        np.save(config['save_center'], self.hash_center.cpu().numpy())

        self.BCEloss = nn.BCELoss().cuda()
        # 初始化为 zeros，不是 randn
        self.Y = torch.zeros(config['num_train'], config['n_class']).float().cuda()
        self.U = torch.randn(config['num_train'], bit).cuda()

        # 可以直接用 eye
        self.label_center = torch.eye(config['n_class'], dtype=torch.float32).cuda()
        self.tanh = nn.Tanh().cuda()

    def forward(self, u1, u2, y, ind, k=0):
        # 缓存当前 batch 的哈希码
        self.U[ind, :] = u2.data

        # 将 y 转为 one-hot 并缓存
        if y.dim() == 1:
            y_onehot = F.one_hot(y, num_classes=self.config['n_class']).float()
        else:
            y_onehot = y
        self.Y[ind, :] = y_onehot

        # 传 one-hot 给后续函数
        return self.cos_pair(u1, y_onehot, ind, k)

    def cos_pair(self, u, y, ind, k):
        """
        :param u: current hash code [B, bit]
        :param y: one-hot label [B, n_class]
        :param ind: global index [B]
        :param k: current epoch
        """
        if k < self.config['epoch_change']:
            pair_loss = 0.0
        else:
            last_u = self.U  # [N, bit]
            last_y = self.Y  # [N, n_class]
            pair_loss = self.moco_pairloss(u, y, last_u, last_y, ind)

        cos_loss = self.cos_eps_loss(u, y, ind)
        Q_loss = (u.abs() - 1.0).pow(2).mean()  # quantization loss

        loss = cos_loss + self.config['beta'] * pair_loss + self.config['lambda'] * Q_loss
        return loss, cos_loss, pair_loss

    def moco_pairloss(self, u, y, last_u, last_y, ind):
        """
        :param u: [B, bit]
        :param y: [B, n_class] one-hot
        :param last_u: [N, bit]
        :param last_y: [N, n_class] one-hot
        :param ind: [B] 当前 batch 的全局索引
        """
        u = F.normalize(u)          # [B, bit]
        last_u = F.normalize(last_u) # [N, bit]

        # 当前 batch 内部的标签相似性
        label_sim = (y @ y.t()) > 0  # [B, B]，同类为 True
        label_sim = label_sim.float()

        # 当前 batch 与历史所有样本的标签相似性
        # y: [B, C], last_y: [N, C] -> [B, N]
        cross_sim = (y @ last_y.t()) > 0  # [B, N]
        cross_sim = cross_sim.float()

        # 当前 batch 的 cos 相似性
        cos_sim = u @ u.t()           # [B, B]
        # 当前与历史的 cos 相似性
        cross_cos = u @ last_u.t()    # [B, N]

        # 只计算正样本对的损失
        # 使用 cross_sim 作为 mask
        pos_mask = cross_sim
        logits = 1/2 * (1 - cross_cos)  # 距离越大，logits 越大
        exp_logits = torch.exp(logits)

        # 对每个样本，正样本 loss
        pos_loss = pos_mask * torch.log(1 + exp_logits)  # [B, N]
        num_pos = pos_mask.sum(dim=1, keepdim=True)      # [B, 1]

        # 避免除以 0
        num_pos = torch.clamp(num_pos, min=1.0)

        # 每个样本的平均正样本 loss
        loss = pos_loss.sum(dim=1, keepdim=True) / num_pos
        loss = loss.mean()  # scalar

        return loss

    def cos_eps_loss(self, u, y, ind):
        K = self.bit
        u_norm = F.normalize(u)                          # [B, bit]
        centers_norm = F.normalize(self.hash_center)     # [n_class, bit]
        cos_sim = u_norm @ centers_norm.t()              # [B, n_class]
        cos_sim = (K ** 0.5) * cos_sim                   # scale

        s = y.float()  # [B, n_class] one-hot label

        p = torch.softmax(cos_sim, dim=1)                # [B, n_class]
        # ✅ 防止 log(0)
        log_p = torch.log(p + 1e-8)
        log_1_minus_p = torch.log(1 - p + 1e-8)
        loss = s * log_p + (1 - s) * log_1_minus_p       # cross entropy
        loss = torch.mean(loss)
        return -loss  # 因为外面要最小化，所以返回负的平均交叉熵
    def get_margin(self):
        # 1. 计算d_min
        L = self.bit
        n_class = self.config['n_class']
        right = (2 ** L) / n_class
        d_min = 0
        d_max = 0
        for j in range(2 * L + 4):
            dim = j
            sum_1 = 0
            sum_2 = 0
            for i in range((dim - 1) // 2 + 1):
                sum_1 += comb(L, i)
            for i in range((dim) // 2 + 1):
                sum_2 += comb(L, i)
            if sum_1 <= right and sum_2 > right:
                d_min = dim
        for i in range(2 * L + 4):
            dim = i
            sum_1 = 0
            sum_2 = 0
            for j in range(dim):
                sum_1 += comb(L, j)
            for j in range(dim - 1):
                sum_2 += comb(L, j)
            if sum_1 >= right and sum_2 < right:
                d_max = dim
        # 2. 计算alpha_neg和alpha_pos
        alpha_neg = L - 2 * d_max
        beta_neg = L - 2 * d_min
        alpha_pos = L
        return alpha_pos, alpha_neg, beta_neg, d_min, d_max

    def generate_center(self, bit, n_class, l):
        hash_centers = np.load(self.config['center_path'])
        self.evaluate_centers(hash_centers)
        hash_centers = hash_centers[l]
        Z = torch.from_numpy(hash_centers).float().cuda()
        return Z
    
    def evaluate_centers(self, H):
        dist = []
        for i in range(H.shape[0]):
            for j in range(i+1, H.shape[0]):
                    TF = np.sum(H[i] != H[j])
                    dist.append(TF)
        dist = np.array(dist)
        st = dist.mean() - dist.var() + dist.min()
        print(f"mean is {dist.mean()}; min is {dist.min()}; var is {dist.var()}; max is {dist.max()}")