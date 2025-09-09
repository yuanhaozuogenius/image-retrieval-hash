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
        "dataset": "cifar10-1",
        "epoch": 120,
        "test_map": 10,
        "device": torch.device("cuda:0"),
        "bit_list": [64],
        "save_path": "save/CSQ_BLIP_3",
        "cifar10_dir": r"D:\Datasets\cifar10",
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
        "enable_caption_sampling": True,  # 开关
        "caption_sampling_k": 3,  # 固定采样数 K
        "caption_prompt": "a photo of",  # 提示词 （caption 前缀）
        "caption_seed": 42,  # 采样随机种子

        # —— 相对路径—— #
        "blip_dir": r"D:\Models\blip2-opt-2.7b",
        "caption_save_path": "./captions.jsonl",  # 生成的jsonl 保存图像生成的caption文本，下次训练时，不需再调模型重复生成，可直接读取
        "fc_path": "./trained_mappers/image_mapper.pth",  # 若没有可注释掉加载
        "med_config": "BLIP/configs/med_config.json",
        "blip_ckpt": "./models/model_base.pth"
    }
    config = config_dataset(config)
    return config


# 可选：BLIP 的标准归一化，当前训练不直接用这个 transform
blip_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


# =========================
# 固定每类采样 K
# =========================
def _sample_fixed_k(cls2items: Dict[int, List], k: int, seed: int = 42):
    random.seed(seed)
    picked: Dict[int, List] = {}
    for c, arr in cls2items.items():
        if len(arr) <= k:
            picked[c] = list(arr)
        else:
            picked[c] = random.sample(arr, k)
    return picked


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

        # 图生文 captioner（仅推理）
        self.captioner = build_image_captioner(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # v256 = “要与文本对齐的语义空间特征向量 feature”（fc1），↔ 文本 256 维
    @torch.no_grad()
    def encode_image_to_256(self, image):
        # 冻结 BLIP，取 CLS 再做 L2 归一化
        vision_embeds = self.model.visual_encoder(image)
        v256 = self.model.vision_proj(vision_embeds[:, 0, :])  # [B, 256]
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

    def forward(self, u, y, ind, config):
        # u 连续 -> tanh 压到 (-1,1)，与中心的二值(-1/1)做 BCE（映射到[0,1]）
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        Q_loss = (u.abs() - 1).pow(2).mean()
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


# —— 对齐损失（发生在 img_adapter(v256) ↔ t256）—— #
class AlignmentLoss(nn.Module):
    def __init__(self, mode="mse"):
        super().__init__()
        self.mode = mode
        self.mse = nn.MSELoss()

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
    # net = config["net"](bit).to(device)
    net = config["net"](bit,
                        blip_ckpt=config.get("blip_ckpt", "./models/model_base.pth"),
                        med_config=config.get("med_config", "BLIP/configs/med_config.json"),
                        fc_path=config.get("fc_path", "./trained_mappers/image_mapper.pth"),
                        text_proj_dim=config.get("text_proj_dim", 256),
                        blip_dir=config.get("blip_dir", config.get("model_dir", r"D:\Models\blip2-opt-2.7b")),
                        ).to(device)
    # 优化器（分组：fc1 / hash head / fc2）
    base_optim = config["optimizer"]
    base_lr = base_optim["optim_params"].get("lr", 1e-5)
    ta_mult = config.get("text_adapter_lr_mult", 0.5)
    params = [
        {"params": net.img_adapter.parameters(), "lr": base_lr},  # fc1
        {"params": net.mapper.parameters(), "lr": base_lr},  # hash head
        {"params": net.text_adapter.parameters(), "lr": base_lr * ta_mult},  # ★ fc2 稍低 lr 更稳
    ]
    optimizer = base_optim["type"](params, **{k: v for k, v in base_optim["optim_params"].items() if k != "lr"})
    # optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    csq_criterion = CSQLoss(config, bit)
    align_criterion = AlignmentLoss(mode=config.get("align_mode", "mse"))
    align_weight = float(config.get("align_weight", 1.0))
    beta = float(config.get("text_anchor_weight", 0.05))
    # 扫描当前 save/算法名/ 文件夹，找到得分最高（mAP 最大）的一组前缀，然后删除其余所有 .pt 和 .npy 文件，只保留那一组
    if "save_path" in config:
        clean_save_dir_keep_best(config["save_path"], config["dataset"])
    Best_mAP = 0

    # 加一个缓存（用 ind 作为键），避免每个 epoch 都重复生成 caption
    caption_cache = {}  # key: int(ind), value: str(caption)
    # 按类缓存（class_id -> List[str]），全训练周期内复用
    class_caption_cache = {}  # e.g., {0: ["a plane ...", ...], 1: [...], ...}
    cap_K = int(config.get("caption_per_class", 5))
    use_class_cache = bool(config.get("caption_cache_by_class", True))

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        net.train()
        train_loss = 0.0
        csq_loss_meter = 0.0
        align_loss_meter = 0.0
        id_loss_meter = 0.0

        for image, label, ind in train_loader:
            image = image.to(device)  # [B,3,H,W]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            label = label.to(device).float()  # [B, n_class]

            # 用 captioner(image) 直接生成对齐文本（带缓存）
            with torch.no_grad():
                y_idx = torch.argmax(label, dim=1).detach().cpu().tolist()  # [B] 每张图的类ID

                if use_class_cache:
                    # 统计本 batch 中，每个类还缺多少代表 caption
                    need_positions = {}  # class_id -> [batch_pos_to_generate]
                    for pos, cls in enumerate(y_idx):
                        cached_list = class_caption_cache.get(cls, [])
                        if len(cached_list) < cap_K:
                            # 为该类收集一些代表位置（最多补到 cap_K 条）
                            lst = need_positions.setdefault(cls, [])
                            if len(lst) < (cap_K - len(cached_list)):
                                lst.append(pos)

                    # 把需要生成的样本拼成一批做一次生成（跨类合批，减少调用次数）
                    if need_positions:
                        gen_pos_all = [p for lst in need_positions.values() for p in lst]
                        imgs_need = image[gen_pos_all]  # Tensor [M,3,H,W]
                        new_caps_all = net.captioner.generate(imgs_need)  # List[str] 长度 M

                        # 按顺序回填到各类的缓存
                        ptr = 0
                        for cls, pos_list in need_positions.items():
                            got = new_caps_all[ptr: ptr + len(pos_list)]
                            ptr += len(pos_list)
                            class_caption_cache.setdefault(cls, []).extend([c.strip() for c in got])

                    #  batch 的每张图选用该类缓存里的一条 caption（随机/轮循均可）
                    captions = []
                    for cls in y_idx:
                        pool = class_caption_cache.get(cls, None)
                        if pool:
                            captions.append(random.choice(pool))
                        else:
                            captions.append("an object")  # 极少数还未生成到时的兜底

                else:
                    # （回退：不启用按类缓存，保持原有逐样本生成逻辑）
                    batch_inds = ind.detach().cpu().tolist()
                    captions = [caption_cache.get(i, None) for i in batch_inds]
                    need_mask = [c is None for c in captions]
                    if any(need_mask):
                        need_idx = [k for k, m in enumerate(need_mask) if m]
                        imgs_need = image[need_idx]
                        new_caps = net.captioner.generate(imgs_need)
                        j = 0
                        for k, m in enumerate(need_mask):
                            if m:
                                captions[k] = new_caps[j]
                                caption_cache[batch_inds[k]] = new_caps[j]
                                j += 1
                    captions = [c if isinstance(c, str) and len(c) > 0 else "an object" for c in captions]

                # 统一编码成 t256
                t256 = net.text_encoder.encode(captions).to(device)
                t256 = F.normalize(t256, dim=-1)

            optimizer.zero_grad()

            # —— 视觉侧：取冻结的 v256；通过 img_adapter（可训练）得到 v_adapt(fc1)—— #
            with torch.no_grad():
                v256 = net.encode_image_to_256(image)  # [B,256] 冻结 BLIP
            v_adapt = net.img_adapter(v256)  # [B,256] ★ 可训练分支
            # 文本侧适配：fc2
            t_adapt = net.text_adapter(t256)  # fc2: [B,256]

            # —— 哈希向量（bit）（可训练）—— #
            u = net.mapper(v_adapt)  # [B,bit]

            # 损失
            align_loss = align_criterion(v_adapt, t_adapt)

            # —— 总损失：CSQ(u, y) + α·align(fc1, fc2) + β·id_loss —— #
            id_loss = F.mse_loss(F.normalize(t_adapt, dim=-1), t256)  # 文本锚定正则：防止 fc2 远离原始 t256 语义
            csq_loss = csq_criterion(u, label, ind, config)
            total_loss = csq_loss + align_weight * align_loss + beta * id_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            csq_loss_meter += csq_loss.item()
            align_loss_meter += align_loss.item()
            id_loss_meter += id_loss.item()

        # 统计并打印 每个 batch 的 loss 会抖动,把整轮（epoch）里所有 batch 的 loss 求平均
        n_iter = len(train_loader)  # batch 数
        train_loss_avg = train_loss / n_iter
        csq_loss_avg = csq_loss_meter / n_iter
        align_loss_avg = align_loss_meter / n_iter
        id_loss_avg = id_loss_meter / n_iter
        lr = optimizer.param_groups[0].get("lr", None)
        lr_str = f"{lr:.1e}" if lr is not None else "NA"
        print(f"{config['info']}[{epoch + 1:>2}/{config['epoch']}][{current_time}] "
              f"bit:{bit}, dataset:{config['dataset']}, "
              f"loss:{train_loss_avg:.3f} "
              f"(csq:{csq_loss_avg:.3f}, align:{align_loss_avg:.3f}, id:{id_loss_avg:.3f}, "
              f"α={align_weight:.2f}, β={beta:.2f}, mode:{config.get('align_mode', 'mse')})")
        # 评估与保存
        if (epoch + 1) % config["test_map"] == 0:
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
