# to_vector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Blip2Model

# ====== 1. 标签转文本 ======
def labels_to_text(ids):
    """
    把类别索引转成文本
    """
    # CIFAR10_NAMES = [
    #     "airplane", "automobile", "bird", "cat", "deer",
    #     "dog", "frog", "horse", "ship", "truck"
    # ]
    # return [CIFAR10_NAMES[i] for i in ids]
    lines = open("data/coco/class_names.txt", "r", encoding="utf-8").read().strip().split("\n")
    names = [line.split()[1] for line in lines]
    return [names[i] for i in ids]


# ====== 2. 文本编码器封装（BLIP-2，本地权重）======
class TextEncoder(nn.Module):
    """
    与原先 BERT 版本保持相同接口：
      - encode(texts: list[str]) -> Tensor[B, proj_dim]
    """
    def __init__(self, model_dir, proj_dim=256, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1) 本地加载 tokenizer / model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        # OPT/GPT2 类 tokenizer 可能没有 pad_token，设置为 eos 以支持 padding
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = Blip2Model.from_pretrained(model_dir)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 2) BLIP-2 文本隐藏维度（不同权重可能不同；从 config 读取最稳妥）
        #    以 OPT-2.7B 为例，hidden_size 通常为 2560
        hidden_size = getattr(self.model.config, "text_config", None)
        if hidden_size is not None:
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            # 兜底：尝试直接读 top-level hidden_size；再不行就先设为 2560
            hidden_size = getattr(self.model.config, "hidden_size", 2560)

        # 3) 投影到你项目里用的维度（默认 256）
        self.proj = nn.Linear(hidden_size, proj_dim)

        # 移动到设备
        self.to(self.device)

    @torch.no_grad()
    def encode(self, texts):
        """
        输入: texts (list[str])，如 ["airplane", "automobile", ...]
        输出: [B, proj_dim] 的归一化向量（可直接用于检索/相似度计算）
        """
        # a) 分词（注意：对 OPT 类 tokenizer 已设置 pad_token）
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt"
        ).to(self.device)

        # b) 前向：让模型返回每层 hidden states
        outputs = self.model.get_text_features(
            **encoded,
            output_hidden_states=True,
            return_dict=True
        )

        # c) 取最后一层隐状态: [B, T, H]
        last_hidden = outputs.hidden_states[-1]

        # d) 用 attention_mask 做均值池化得到句向量: [B, H]
        mask = encoded["attention_mask"].unsqueeze(-1)         # [B, T, 1]
        summed = (last_hidden * mask).sum(dim=1)               # [B, H]
        counts = mask.sum(dim=1).clamp(min=1)                  # [B, 1]
        sent_embed = summed / counts                           # [B, H]

        # e) 投影到 proj_dim 并做 L2 归一化
        proj_embed = self.proj(sent_embed)                     # [B, proj_dim]
        return F.normalize(proj_embed, dim=-1)                 # [B, proj_dim]


# ====== 3. 工厂函数（保持原名与签名，主程序可零改动调用）======
def build_text_encoder(
    model_dir=r"D:\Models\blip2-opt-2.7b",
    proj_dim=256,
    device="cuda"
):
    return TextEncoder(model_dir=model_dir, proj_dim=proj_dim, device=device)
