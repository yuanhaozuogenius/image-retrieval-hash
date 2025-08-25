import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def save_image(data, path):
    img = np.reshape(data, (3, 32, 32)).transpose(1, 2, 0)  # CHW → HWC
    img = Image.fromarray(img)
    img.save(path)

def save_txt(indices, labels, prefix, txt_path):
    with open(txt_path, 'w') as f:
        for i, idx in enumerate(indices):
            filename = f'{prefix}/image_{idx:05d}.png'  # ⬅️ 注意这里是 idx 而不是累加的 start_idx
            label = labels[idx]
            f.write(f'{filename}\t{label}\n')


def prepare():
    cifar_dir = '../data/cifar_init/cifar-10-batches-py'
    save_dir = '../data/cifar10'
    os.makedirs(save_dir, exist_ok=True)

    all_data = []
    all_labels = []

    # 1. 加载所有数据
    for i in range(1, 6):
        batch = unpickle(os.path.join(cifar_dir, f'data_batch_{i}'))
        all_data.extend(batch[b'data'])
        all_labels.extend(batch[b'labels'])

    test_batch = unpickle(os.path.join(cifar_dir, 'test_batch'))
    all_data.extend(test_batch[b'data'])
    all_labels.extend(test_batch[b'labels'])

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    total = len(all_labels)
    rng = np.random.RandomState(42)
    indices = rng.permutation(total)

    # 2. 划分
    test_idx = indices[:1000]
    database_idx = indices[1000:]
    train_idx = database_idx[:5000]
    valid_idx = database_idx[5000:10000]
    subtrain_idx = database_idx[10000:11000]

    # 3. 保存图像
    print("🖼 Saving images...")
    for i, idx in enumerate(indices):
        img_data = all_data[idx]
        filename = f'image_{i:05d}.png'
        save_image(img_data, os.path.join(save_dir, filename))
    print("✅ 图像保存完成。")

    # 4. 保存 txt
    save_txt(train_idx, all_labels, 'cifar10', os.path.join(save_dir, 'train.txt'))
    save_txt(test_idx, all_labels, 'cifar10', os.path.join(save_dir, 'test.txt'))
    save_txt(valid_idx, all_labels, 'cifar10', os.path.join(save_dir, 'valid.txt'))
    save_txt(subtrain_idx, all_labels, 'cifar10', os.path.join(save_dir, 'subtrain.txt'))
    save_txt(database_idx, all_labels, 'cifar10', os.path.join(save_dir, 'database.txt'))

    print('✅ 完成划分：test=1000, database=59000 (含train/valid/subtrain)')


if __name__ == '__main__':
    prepare()
