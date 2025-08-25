from lavis.models import load_model_and_preprocess
from transformers import BertTokenizer
from transformers import Blip2ForConditionalGeneration
from PIL import Image
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载图像
raw_image = Image.open("data/image.png").convert("RGB")

# ✅ 正确指定本地模型路径
model, vis_processors, text_processors = load_model_and_preprocess(
    name="blip2",
    model_type="pretrain",
    is_eval=True,
    device="cuda"
)


# ==== 2. 重新加载权重并去掉冲突 key ====
url = model.default_config.pretrained.get("pretrain")  # 从配置获取官方下载地址
ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=False)

# 删除导致词表 mismatch 的 MLM 头权重
ckpt.pop("Qformer.cls.predictions.bias", None)
ckpt.pop("Qformer.cls.predictions.weight", None)

# 加载权重
model.load_state_dict(ckpt, strict=False)

# ==== 3. 使用官方 BERT tokenizer ====
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # 图像处理
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# 文本处理
# CIFAR-10 标签
labels = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

processed_texts = [text_processors["eval"](label) for label in labels]

text_inputs = tokenizer(
    processed_texts,
    padding="max_length",
    truncation=True,
    max_length=32,
    return_tensors="pt"
).to(device)

if "token_type_ids" in text_inputs:
    text_inputs.pop("token_type_ids")

# ==== 5. 获取文本 fc2 特征 ====
with torch.no_grad():
    qformer_outputs = model.Qformer.bert(
        input_ids=text_inputs.input_ids,
        attention_mask=text_inputs.attention_mask,
        return_dict=True
    )
    cls_feat = qformer_outputs.last_hidden_state[:, 0, :]
    text_fc2 = model.text_proj(cls_feat)

print("文本 fc2 形状:", text_fc2.shape)
print("第一个标签 fc2 前 5 维:", text_fc2[0, :5])


# 生成标题
# caption = model.generate({"image": image})
# print("📝 Generated Caption:", caption)