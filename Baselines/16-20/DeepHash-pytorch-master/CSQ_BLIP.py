import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 可选：将 BLIP 子目录添加用于导入模型
blip_path = os.path.join(ROOT, 'BLIP')
if blip_path not in sys.path:
    sys.path.append(blip_path)

from utils.tools import *

from BLIP.models.blip_itm import blip_itm
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
import time

torch.multiprocessing.set_sharing_strategy('file_system')


def build_blip_net(bit):
    return BLIP_HashWrapper(bit)


def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ_BLIP]",
        "resize_size": 224,
        "crop_size": 224,
        "batch_size": 64,
        "net": build_blip_net,
        "dataset": "cifar10-1",
        "epoch": 150,
        "test_map": 10,
        "device": torch.device("cuda:0"),
        "bit_list": [64],# 哈希码位数
        "save_path": "save/CSQ_BLIP",
    }
    config = config_dataset(config)
    return config


blip_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


class FeatureMapper(nn.Module):
    def __init__(self, input_dim=256, output_dim=64):
        super(FeatureMapper, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)


class BLIP_HashWrapper(nn.Module):
    def __init__(self, bit, blip_ckpt='./models/model_base.pth', med_config='BLIP/configs/med_config.json',
                 fc_path='./trained_mappers/image_mapper.pth'):
        super().__init__()
        self.model = blip_itm(pretrained=blip_ckpt, med_config=med_config, image_size=224, vit='base')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.mapper = FeatureMapper(256, bit)
        self.mapper.load_state_dict(torch.load(fc_path, map_location=torch.device('cuda')))
        # self.mapper.load_state_dict(torch.load(fc_path, map_location='cpu'))

    def forward(self, image):
        with torch.no_grad():
            vision_embeds = self.model.visual_encoder(image)
            feat = self.model.vision_proj(vision_embeds[:, 0, :])  # [B, 256]
            feat = F.normalize(feat, dim=-1)
        return self.mapper(feat)  # [B, bit]


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y, ind, config):
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


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = CSQLoss(config, bit)
    # 扫描当前 save/算法名/ 文件夹，找到得分最高（mAP 最大）的一组前缀，然后删除其余所有 .pt 和 .npy 文件，只保留那一组
    if "save_path" in config:
        clean_save_dir_keep_best(config["save_path"], config["dataset"])
    Best_mAP = 0

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            u = net(image)
            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            tst_binary, tst_label = compute_result(test_loader, net, device=device)
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            if mAP > Best_mAP:
                Best_mAP = mAP
                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    save_path = config["save_path"]
                    # 替换 "-" → "_"，确保正则或文件名不会误解析
                    dataset_tag = config["dataset"].replace("-", "_")
                    # 格式化 MAP 保留固定小数位，避免路径长度混乱
                    score_str = f"{mAP:.10f}"
                    filename_prefix = f"{dataset_tag}-{score_str}"
                    # 保存模型及中间文件
                    np.save(os.path.join(save_path, f"{filename_prefix}-trn_binary.npy"), trn_binary.numpy())
                    np.save(os.path.join(save_path, f"{filename_prefix}-tst_binary.npy"), tst_binary.numpy())
                    np.save(os.path.join(save_path, f"{filename_prefix}-trn_label.npy"), trn_label.numpy())
                    np.save(os.path.join(save_path, f"{filename_prefix}-tst_label.npy"), tst_label.numpy())
                    torch.save(net.state_dict(), os.path.join(save_path, f"{filename_prefix}-model.pt"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
    print(">>> script reached end of file")
