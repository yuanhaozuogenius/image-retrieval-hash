from collections import Counter

file = "data/coco/train.txt"
classes = []

with open(file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        # 解析多标签标识
        lbls = [int(x) for x in parts[1:]]
        for i, v in enumerate(lbls):
            if v == 1:  # 如果该图片属于类别 i
                classes.append(i)

cnt = Counter(classes)
print("类别70样本数:", cnt[70])
print("类别79样本数:", cnt[79])
