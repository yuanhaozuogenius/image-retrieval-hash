# SPRCH
This is the source code implementation for Self-paced Relational Contrastive Hashing for Large-scale Image Retrieval. We will release it after publication.

## 脚本运行
在main目录下运行run_coco.sh, 路径拼接逻辑在dataloader.py中

另外目前设置的是 transforms.Resize(256) 后续可以调整


## command
 `
 python main.py --data_path D:/Datasets --data_name cifar100 --data_class 100 --batchSize 64 --binary_bits 64 --epochs 200 --lr 1e-5
 `

  `
 python main.py --data_path D:/Datasets/cifar10 --data_name cifar10 --data_class 10 --batchSize 64 --binary_bits 64 --epochs 200 --lr 1e-5
 `