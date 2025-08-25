import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# ===== 本地 BLIP 代码路径 =====
import sys

sys.path.insert(0, './BLIP')
from BLIP.models.blip_itm import blip_itm

# ====== 参数设置 ======
image_list_txt = "./data/your_image_list.txt"
text_list_txt = "./data/your_text_list.txt"

pretrained_ckpt = './models/model_base.pth'
# 模型下载地址https://github.com/salesforce/BLIP?tab=readme-ov-file
med_config_file = 'BLIP/configs/med_config.json'

batch_size = 32
epochs = 10
lr = 1e-4

# ====== 设备 ======
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 模型加载 ======
model = blip_itm(pretrained=pretrained_ckpt, med_config=med_config_file, image_size=224, vit='base')
model = model.to(device).eval()

# ====== 图像预处理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])


# ====== 额外的全连接层（可训练） ======
# 作用：BLIP 提供的特征可能还不够“语义对齐”；用小网络（只1层FC）微调，让图像和文本特征更相似
class FeatureMapper(nn.Module):
    def __init__(self, input_dim=256, output_dim=64):
        super().__init__()
        # 定义一个wx+b的线性变换全连接层，反向传播时，误差会从损失函数传回fc(x)，并更新w,b
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 把输出向量的模长变成 1；方向保持不变；
        return F.normalize(self.fc(x), dim=-1)


image_mapper = FeatureMapper().to(device)
text_mapper = FeatureMapper().to(device)

# ====== 优化器 ======
# 没有训练 BLIP 本体（visual_encoder/text_encoder），也没有更新 tokenizer， 所以： BLIP 的表示是固定不变的；
# 只是学了一个小型映射器（即1层全连接层）
# Adam:使用自适应梯度优化器，学习率动态调整，适合稀疏梯度和小模型
optimizer = optim.Adam(list(image_mapper.parameters()) + list(text_mapper.parameters()), lr=lr)

# ====== 加载数据 ======
with open(image_list_txt, 'r', encoding='utf-8') as f:
    image_paths = [line.strip() for line in f if line.strip()]

with open(text_list_txt, 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

assert len(image_paths) == len(texts), "图片数量和文本数量不一致"


# ====== 特征提取函数 ======
def extract_image_feature(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vision_embeds = model.visual_encoder(image_tensor)
        feat = model.vision_proj(vision_embeds[:, 0, :])
        feat = F.normalize(feat, dim=-1)
    return feat


def extract_text_feature(text):
    text_input = model.tokenizer(
        text,
        padding='longest',
        truncation=True,
        max_length=200,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_output = model.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            return_dict=True,
            mode='text'
        )
        feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
        feat = F.normalize(feat, dim=-1)
    return feat


# ====== 定义损失函数（可以很方便切换） ======
# 1.定义 PyTorch 的余弦损失函数模块;
# 2.创建全是1的标签向量，表示每对图文是“正样本”;
# 3.对每对图文计算余弦损失，推动相似度更高
def compute_loss(img_proj, txt_proj):
    loss_fn = nn.CosineEmbeddingLoss()
    targets = torch.ones(img_proj.size(0)).to(device)
    loss = loss_fn(img_proj, txt_proj, targets)
    return loss


# ====== 训练流程 ======
def train():
    print("\nStart training mapper layers...")
    for epoch in range(epochs):
        model.eval()
        image_mapper.train()
        text_mapper.train()

        epoch_loss = 0.0
        for idx in tqdm(range(0, len(image_paths), batch_size)):
            batch_images = image_paths[idx: idx + batch_size]
            batch_texts = texts[idx: idx + batch_size]

            image_feats = []
            text_feats = []
            for img_path, txt in zip(batch_images, batch_texts):
                try:
                    img_feat = extract_image_feature(img_path)
                    txt_feat = extract_text_feature(txt)
                    image_feats.append(img_feat)
                    text_feats.append(txt_feat)
                except Exception as e:
                    print(f"[!] 出错跳过: {img_path}, 错误: {e}")
                    continue

            if len(image_feats) == 0:
                continue

            image_feats = torch.cat(image_feats, dim=0)
            text_feats = torch.cat(text_feats, dim=0)

            optimizer.zero_grad()

            img_proj = image_mapper(image_feats)
            txt_proj = text_mapper(text_feats)

            loss = compute_loss(img_proj, txt_proj)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}")

    print("✅ 训练完成！")


# ====== 运行训练 ======
if __name__ == "__main__":
    train()

    # 保存模型
    save_dir = "./trained_mappers"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(image_mapper.state_dict(), os.path.join(save_dir, "image_mapper.pth"))
    torch.save(text_mapper.state_dict(), os.path.join(save_dir, "text_mapper.pth"))
    print("✅ 映射器保存完成！")
