# to_label.py —  captioner for CSQ_BLIP_3
# - 输入：一个 batch 的图像张量 [B,3,H,W]
# - 输出：等长的 caption 列表 List[str]
# - 本地权重、eval()、no_grad()、批量生成

from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# --- helpers: Tensor[B,3,H,W] -> List[PIL.Image] ---
@torch.no_grad()
def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    """将标准化/任意范围的图像张量转到 [0,255] uint8（CPU）。"""
    x = images.detach().float().cpu()
    if x.ndim != 4 or x.size(1) != 3:
        raise ValueError("expect [B,3,H,W] tensor")
    mn, mx = float(x.min()), float(x.max())
    if 0.0 <= mn <= mx <= 1.0:
        x = (x * 255.0).clamp(0, 255)
    else:
        # 经验反归一化（ImageNet 均值方差）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        x = (x * std + mean).clamp(0, 1) * 255.0
    return x.round().byte()

@torch.no_grad()
def _tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
    arr = _to_uint8(images).permute(0, 2, 3, 1).numpy()  # [B,H,W,3]
    return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]


# --- minimal captioner ---
class Blip2Captioner:
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        num_beams: int = 3,# num_beams  beam search的宽度, 数值越大，探索备选序列更多、句子通常更合理，但速度更慢、显存更大；训练循环里建议 1–3
        max_new_tokens: int = 32,# max_new_tokens：从起始 token（或提示词之后）最多生成多少个新 token。越大句子越长、时间越久；做跨模态对齐只需概要语义，16–32 已够用
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_dir, local_files_only=True,use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
        self.model.to(self.device).eval()
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)

    @torch.no_grad()
    def generate(
            self,
            images: torch.Tensor,
            prompt: Optional[str] = None,  # 支持可选提示词
            return_scores: bool = False,  # 需要返回每条 caption 的分数
    ) -> Tuple[List[str], Optional[torch.Tensor]]:
        """
        输入 batch 张量，返回：
          - captions: List[str]（长度 = B）
          - scores:  Tensor[B] 或 None（当 return_scores=True 且可用时）
        """
        pil_list = _tensor_to_pil(images)
        if prompt is not None and len(prompt) > 0:
            inputs = self.processor(images=pil_list, text=[prompt] * len(pil_list), return_tensors="pt")
        else:
            inputs = self.processor(images=pil_list, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = dict(num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)

        if return_scores:
            # 要返回分数 → 使用 dict 输出，并开启 output_scores
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
                return_dict_in_generate=True,
                output_scores=True,
                num_return_sequences=self.num_beams,  # 每张图返回 top-B 条
            )
            out_ids = outputs.sequences
            captions = self.processor.batch_decode(out_ids, skip_special_tokens=True)
            # 优先用 sequences_scores（beam search 时稳定可用）
            scores = getattr(outputs, "sequences_scores", None)
            if scores is None:
                # 若为 greedy 或特殊配置可能没有 sequences_scores，这里兜底给 0 分
                scores = torch.zeros(len(captions), device=self.device)
            return [t.strip() for t in captions], scores.detach().float()
        else:
            out_ids = self.model.generate(**inputs, **gen_kwargs)
            captions = self.processor.batch_decode(out_ids, skip_special_tokens=True)
            return [t.strip() for t in captions], None



# 工厂函数
def build_image_captioner(model_dir: str, device: str = "cuda", num_beams: int = 3, max_new_tokens: int = 32) -> Blip2Captioner:
    return Blip2Captioner(model_dir=model_dir, device=device, num_beams=num_beams, max_new_tokens=max_new_tokens)

