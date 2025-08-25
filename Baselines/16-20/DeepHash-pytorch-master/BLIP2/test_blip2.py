from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载图像
raw_image = Image.open("./data/image.png").convert("RGB")

# ✅ 正确指定本地模型路径
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="caption_coco_opt2.7b",  # 使用的模型
    device=device,
)


# 图像处理
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# 生成标题
caption = model.generate({"image": image})
print("📝 Generated Caption:", caption)

