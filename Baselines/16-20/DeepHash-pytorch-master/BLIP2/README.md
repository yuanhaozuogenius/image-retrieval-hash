# BLIP-2 模型测试与配置指南

## 🔧 环境配置

请先根据以下命令安装项目依赖：

```bash
conda env create -f environment.yml
```

## 🚀 运行示例

使用 Hugging Face 镜像源运行模型测试脚本：

```
HF_ENDPOINT=https://hf-mirror.com python test_blip2.py
```

## 📂 数据存储位置

默认数据存储目录为：

```
./data/
```

## 🧠 模型配置说明

您可以通过修改 `test_blip2.py` 文件中的 `name` 与 `model_type` 参数来自定义加载的模型。以下是支持的模型名称及其对应类型：

| 模型名称 (`name`) | 支持的模型类型 (`model_type`)                                |
| ----------------- | ------------------------------------------------------------ |
| `blip2_opt`       | `pretrain_opt2.7b` `caption_coco_opt2.7b` `pretrain_opt6.7b` `caption_coco_opt6.7b` |
| `blip2_t5`        | `pretrain_flant5xl` `caption_coco_flant5xl` `pretrain_flant5xxl` |
| `blip2`           | `pretrain` `coco`                                            |

⚠️ 请确保 `name` 和 `model_type` 的组合是兼容的，否则将导致模型加载失败。