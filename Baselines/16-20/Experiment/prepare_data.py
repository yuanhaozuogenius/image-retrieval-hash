import os
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, Caltech101, Food101, OxfordIIITPet
from PIL import Image

# ==== 配置参数🔁 修改为你想测试的数据集名（小写） ====
dataset_name = "cifar10"  #

save_dir = "./data"
img_dir = os.path.join(save_dir, dataset_name)
img_txt_path = os.path.join(save_dir, "your_image_list.txt")
text_txt_path = os.path.join(save_dir, "your_text_list.txt")
N = 200  # 可调数量
resize_size = (224, 224)

# ==== 数据集映射（支持的）====
dataset_map = {
    "cifar10": lambda: CIFAR10(root=save_dir, train=True, download=True),
    "cifar100": lambda: CIFAR100(root=save_dir, train=True, download=True),
    "stl10": lambda: STL10(root=save_dir, split="train", download=True),
    "caltech101": lambda: Caltech101(root=save_dir, download=True),
    "food101": lambda: Food101(root=save_dir, split="train", download=True),
    "oxfordpets": lambda: OxfordIIITPet(root=save_dir, split="trainval", download=True),
}

# ==== 获取标签名 ====
def get_label_name(dataset, label):
    if hasattr(dataset, 'classes'):
        return dataset.classes[label]
    elif hasattr(dataset, 'cat_to_name'):
        return dataset.cat_to_name[label]
    else:
        return str(label)

# ==== 加载数据集 ====
if dataset_name not in dataset_map:
    raise ValueError(f"❌ 不支持的数据集: {dataset_name}")
dataset = dataset_map[dataset_name]()
print(f"📦 成功加载数据集: {dataset_name}")

# ==== 图像变换（统一 224×224）====
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor()
])

# ==== 创建输出目录 ====
os.makedirs(img_dir, exist_ok=True)

# ==== 构建图文对 ====
with open(img_txt_path, 'w', encoding='utf-8') as f_img, open(text_txt_path, 'w', encoding='utf-8') as f_txt:
    for i in range(N):
        image, label = dataset[i]
        label_text = f"This is a photo of a {get_label_name(dataset, label)}"
        filename = f"{i:05d}.jpg"
        image_path = os.path.join(img_dir, filename)

        image = transforms.ToPILImage()(transform(image))
        image.save(image_path)

        f_img.write(image_path.replace("\\", "/") + "\n")
        f_txt.write(label_text + "\n")

print(f"✅ 数据集 {dataset_name} 预处理完成，已生成 {N} 个图文对")
