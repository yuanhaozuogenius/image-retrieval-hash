'''
由Blip2的 get_image_features 实现
https://huggingface.co/docs/transformers/model_doc/blip-2

如果只是要“抽图像向量”：用 Blip2Model.get_image_features(...)（表征式）
'''
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2Model

# 本地模型目录（你已经下载好的）
MODEL_DIR = r"D:\Models\blip2-opt-2.7b"

# 1) 加载本地模型与处理器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Blip2Model.from_pretrained(MODEL_DIR).to(device).eval()
processor = AutoProcessor.from_pretrained(MODEL_DIR)

# 2) 读图（本地文件；如果你仍要 requests.get 也可以，只是本地更稳）
image = Image.open(r"imgs/dogs.jpg").convert("RGB")

# 3) 处理成模型输入
inputs = processor(images=image, return_tensors="pt").to(device)

# 4) 抽取图像特征（官方推荐接口）
with torch.no_grad():
    image_outputs = model.get_image_features(**inputs)  # 通常直接是张量或包含特征的输出对象

# 5) 看形状（简单确认）
try:
    print("image_features shape:", image_outputs.shape)
except AttributeError:
    # 有些版本会把特征放在属性里（稳妥兜底）
    feats = getattr(image_outputs, "image_embeds", None) or getattr(image_outputs, "pooler_output", None)
    if feats is None:
        raise RuntimeError(f"Unexpected output type: {type(image_outputs)}; keys = {getattr(image_outputs, 'keys', lambda: [])()}")
    print("image_features shape:", feats.shape)
