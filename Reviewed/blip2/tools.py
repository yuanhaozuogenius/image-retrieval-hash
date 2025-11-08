from typing import Dict, List, Tuple, Optional
import os
import json

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