# pip install accelerate
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from pathlib import Path


# processor = Blip2Processor.from_pretrained(r"/Baselines/Reviewed/blip2/weight")
# model = Blip2ForConditionalGeneration.from_pretrained(r"/Baselines/Reviewed/blip2/weight").to("cuda")
model_dir = Path(__file__).resolve().parent / "weight"

processor = Blip2Processor.from_pretrained(model_dir, local_files_only=True)
model = Blip2ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).to("cuda")


# raw_image = Image.open("/Baselines/Reviewed/blip2/imgs/201707041538117hOYR7hOYR.jpg").convert('RGB')
raw_image = Image.open("imgs/201707041538117hOYR7hOYR.jpg").convert("RGB")


inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())
