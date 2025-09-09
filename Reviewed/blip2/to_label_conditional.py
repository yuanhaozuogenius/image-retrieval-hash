'''
BLIP-2 Model for generating text given an image and an optional text prompt. The model consists of a vision encoder, Querying Transformer (Q-Former) and a language model.
BLIP-2 模型用于根据图像和可选的文本提示生成文本。该模型由一个视觉编码器、查询 Transformer（Q-Former）和语言模型组成。

One can optionally pass input_ids to the model, which serve as a text prompt, to make the language model continue the prompt. Otherwise, the language model starts generating text from the [BOS] (beginning-of-sequence) token.
可以选择将 input_ids 传递给模型，它作为文本提示，使语言模型继续提示。否则，语言模型将从[BOS]（序列开始）标记开始生成文本。
https://huggingface.co/docs/transformers/model_doc/blip-2

如果是要“生 caption/标签文案”：用 Blip2ForConditionalGeneration.generate(...)（生成式）
'''
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


model_dir = r"D:\Models\blip2-opt-2.7b"

processor = Blip2Processor.from_pretrained(model_dir, local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).to("cuda")

# raw_image = Image.open("/Baselines/Reviewed/blip2/imgs/201707041538117hOYR7hOYR.jpg").convert('RGB')
raw_image = Image.open("imgs/dogs.jpg").convert("RGB")
# raw_image = Image.open("imgs/201707041538117hOYR7hOYR.jpg").convert("RGB")


inputs = processor(raw_image, return_tensors="pt").to("cuda")

generated_ids  = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
