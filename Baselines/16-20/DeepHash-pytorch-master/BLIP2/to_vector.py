# to_vector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from transformers import AutoTokenizer, Blip2Model
from utils.tools import *
from pathlib import Path


# def labels_to_text(ids):
#     """
#     把类别索引转成文本
#     """
#     CIFAR10_NAMES = [
#         "airplane", "automobile", "bird", "cat", "deer",
#         "dog", "frog", "horse", "ship", "truck"
#     ]
#     return [CIFAR10_NAMES[i] for i in ids]

# ====== 2. 文本编码器封装（BLIP-2，本地权重）======
class TextEncoder(torch.nn.Module):
    """
    简要说明：
    - 负责把若干文本（prompts/captions）编码为定长向量（维度 self.proj.out_features）。
    - 设计为可批量输入，避免多次前向开销。
    """

    def __init__(self, model_dir, proj_dim, max_length, device):
        super().__init__()
        self.max_length = max_length
        self.device = torch.device(device)
        self.model = Blip2Model.from_pretrained(model_dir).to(self.device)
        self.model.eval()  # PyTorch 模型的“评估模式； 设为推理阶段（inference），不做训练时的随机性或统计更新。
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # 本地加载 tokenizer / model

        # 2) BLIP-2 文本隐藏维度（不同权重可能不同；从 config 读取最稳妥）
        #    以 OPT-2.7B 为例，hidden_size 通常为 2560
        hidden_size = getattr(self.model.config, "text_config", None)
        if hidden_size is not None:
            hidden_size = getattr(self.model.config.text_config, "hidden_size", None)
        if hidden_size is None:
            # 兜底：尝试直接读 top-level hidden_size；再不行就先设为 2560
            hidden_size = getattr(self.model.config, "hidden_size", 2560)
        self.proj = nn.Linear(hidden_size, proj_dim).to(self.device)  # 3) 投影到你项目里用的维度（默认 256）

    @torch.no_grad()
    def encode(self, texts: Union[str, List[str]], batch_size: int = 64):
        """
        功能：将 N 条文本编码为 [N, D] 的向量（已归一化）。
        参数：
            texts: 单条或多条文本（str 或 List[str]）
            batch_size: 编码时的微批大小，控制速度/显存
        返回：
            feats: torch.FloatTensor，形状 [N, D]，已 L2 归一化（与 CSQ_BLIP_4 下游对齐）
        说明：
            - 使用 model.get_text_features()，直接得到 [B, H] 的 pooled 表征，避免 last_hidden_state / hidden_states 的额外开销。
            - 不打开 output_hidden_states、不打开 return_dict，速度/显存更优。
        """
        N = len(texts)
        out_list = []
        for s in range(0, N, batch_size):  # 批处理循环：一次只送入 batch_size 条文本（例如 64 条）进行编码
            sub = texts[s:s + batch_size]
            inputs = self.tokenizer(
                sub,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # 前向：让模型返回每层 hidden states
            outputs = self.model.get_text_features(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            # c) 取最后一层隐状态: [B, T, H]
            last_hidden = outputs.hidden_states[-1]
            # d) 用 attention_mask 做均值池化得到句向量: [B, H]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)  # [B, H]
            counts = mask.sum(dim=1).clamp(min=1)  # [B, 1]
            sent_embed = summed / counts

            proj_embed = self.proj(sent_embed)  # [B, proj_dim] 投影到 proj_dim
            proj_bd = F.normalize(proj_embed, dim=-1)  # 归一化，便于后续余弦对比/对齐
            out_list.append(proj_bd)

        return torch.cat(out_list, dim=0)  # 拼成单一张量返回（用于 bank）


# ====== 3. 工厂函数（保持原名与签名，主程序可零改动调用）======
def build_text_encoder(
        model_dir=r"D:\Models\blip2-opt-2.7b",
        proj_dim=256,
        max_length=64,
        device="cuda"
):
    return TextEncoder(model_dir=model_dir, proj_dim=proj_dim, device=device, max_length=max_length)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    cache_path = str(ROOT / "data" / "coco" / "captions.jsonl")
    class2caps = load_class_captions(cache_path)
    # 每类取第 1 条 caption（若缺失就用类名）
    captions = [
        (caps[0] if isinstance(caps, list) and len(caps) > 0 else f"class {c}")
        for c, caps in sorted(class2caps.items())
    ]
    text_encoder = build_text_encoder()
    t256 = text_encoder.encode(captions)  # [B,256]
    print("t256.shape:", t256.shape)
