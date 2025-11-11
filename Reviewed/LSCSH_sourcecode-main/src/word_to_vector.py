import pickle
from sentence_transformers import SentenceTransformer
import torch

# 定义COCO类别
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
model_dir =  r"D:/Models/bert-base-uncased"  # Sentence-Transformers 本地目录

# 加载BERT编码器
model = SentenceTransformer(model_dir)  # 输出维度768

# 编码
embeddings = model.encode(coco_classes, convert_to_tensor=True)

# 保存为pkl
save_path="../data/coco/coco_bert768_word2vec.pkl"    # 要求 word_embeddings 输出维度是 [64, 80, 768]
with open(save_path, 'wb') as f:
    pickle.dump({
        "class_names": coco_classes,
        "word_embeddings": embeddings.cpu()
    }, f)

print(f"Saved word2vec to {save_path}")
print(f"Shape: {embeddings.shape}")
