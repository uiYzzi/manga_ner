# 漫画标题命名实体识别 (Manga NER)

一个基于深度学习的漫画标题命名实体识别系统，能够从漫画文件名中提取关键信息如作品名、作者、卷号等。

## 🎯 项目概述

本项目使用预训练的多语言BERT模型（mBERT）对漫画文件名进行命名实体识别（NER），自动提取以下实体：
- **发布组** (GROUP)：汉化组、扫图组或发布组织
- **作品标题** (TITLE)：漫画的核心名称
- **卷/话数** (VOLUME/CHAPTER)：漫画的卷号或话数信息
- **作者** (AUTHOR)：漫画作者姓名
- **元数据标签** (TAG)：版本、来源、活动等描述信息
- **分隔符与容器** (O)：包括空格、`_`、`-`、`~`等分隔符，以及所有类型的括号
- **文件后缀** (O)：`.zip`、`.cbz`、`.rar`等文件扩展名

## 🏗️ 项目结构

```
manga_ner/
├── data/                          # 数据目录
│   ├── chunked_annotations.jsonl  # 原始标注数据（文本块格式）
│   ├── train.txt                  # 训练集（IOB2格式）
│   ├── validation.txt             # 验证集（IOB2格式）
│   └── test.txt                   # 测试集（IOB2格式）
├── src/                           # 源代码目录
│   ├── convert_to_iob2.py         # 数据格式转换工具
│   ├── train.py                   # 模型训练脚本
│   ├── inference.py               # 推理预测脚本
│   └── validate_data.py           # 数据验证工具
├── manga_ner_model/               # 训练好的模型保存目录
├── pyproject.toml                 # 项目依赖配置
└── README.md                      # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch
- Transformers库
- 推荐使用GPU加速（支持Apple Silicon的MPS）

### 安装依赖

使用uv进行依赖管理：

```bash
pip install uv  # 如果还未安装uv
uv sync        # 安装所有依赖
```

或者使用pip：

```bash
pip install -r requirements.txt
```

### 数据准备

1. **准备原始标注数据**
   将你的标注数据保存为`data/chunked_annotations.jsonl`，格式为每行一个JSON对象：
   ```json
   {"text": "【漫画】一拳超人 第15卷", "entities": [{"start": 4, "end": 8, "label": "TITLE"}, {"start": 10, "end": 13, "label": "VOLUME"}]}
   ```

2. **转换数据格式**
   运行数据转换脚本将原始数据转换为IOB2格式：
   ```bash
   python src/convert_to_iob2.py data/chunked_annotations.jsonl
   ```
   这将自动生成`train.txt`、`validation.txt`和`test.txt`文件。

### 模型训练

运行训练脚本：

```bash
python src/train.py
```

训练过程将：
- 自动加载预训练的mBERT模型
- 使用训练数据进行微调
- 在验证集上评估性能
- 将最佳模型保存到`manga_ner_model/`目录

### 使用模型进行预测

训练完成后，使用推理脚本解析漫画文件名：

```bash
python src/inference.py
```

或者在Python代码中使用：

```python
from transformers import pipeline

# 加载训练好的模型
ner_pipeline = pipeline("token-classification", model="./manga_ner_model/best_model")

# 进行预测
result = ner_pipeline("【漫画】进击的巨人 第28卷 谏山创")
print(result)
```

## 📊 数据格式说明

### IOB2格式

训练和测试数据使用标准的IOB2格式：

```
[ B-GROUP
澄 I-GROUP
空 I-GROUP
学 I-GROUP
园 I-GROUP
] I-GROUP
空 B-TITLE
之 I-TITLE
空 I-TITLE
第 B-VOLUME
1 I-VOLUME
卷 I-VOLUME
新 B-TAG
版 I-TAG
.zip O
```

### 支持的实体标签

项目目前支持以下7种实体类型：

- **发布组 (`GROUP`)**: 汉化组、扫图组或发布组织
- **作品标题 (`TITLE`)**: 漫画的核心名称
- **卷/话数 (`VOLUME`/`CHAPTER`)**: 漫画的卷号或话数信息
- **作者 (`AUTHOR`)**: 漫画作者姓名
- **元数据标签 (`TAG`)**: 版本、来源、活动等描述信息
- **分隔符与容器 (`O`)**: 包括空格、`_`、`-`、`~`等分隔符，以及所有类型的括号如`[]`、`()`、`【】`、`{}`、`「」`
- **文件后缀 (`O`)**: `.zip`、`.cbz`、`.rar`等文件扩展名

#### 标签映射表
| 实体类型 | B-标签 | I-标签 | 说明 |
|---------|--------|--------|------|
| 发布组 | B-GROUP | I-GROUP | 汉化组、发布组织 |
| 作品标题 | B-TITLE | I-TITLE | 漫画核心名称 |
| 卷/话数 | B-VOLUME | I-VOLUME | 卷号或话数 |
| 作者 | B-AUTHOR | I-AUTHOR | 作者姓名 |
| 元数据标签 | B-TAG | I-TAG | 版本、来源等 |
| 分隔符/后缀 | O | O | 括号、空格、文件扩展名等 |

## 🔧 配置参数

在`src/train.py`中可以调整以下参数：

- `MODEL_NAME`: 预训练模型名称（默认：`distilbert-base-multilingual-cased`）
- `DATA_PATH`: 数据文件路径（默认：`./data/`）
- `OUTPUT_DIR`: 模型保存路径（默认：`./manga_ner_model`）
- 训练参数：学习率、批次大小、训练轮数等

## 📈 性能评估

模型在验证集上的性能指标包括：
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1-score）

评估结果将在训练过程中自动输出。

## 🛠️ 开发指南

### 添加新的实体类型

1. 在数据标注时添加新的实体标签
2. 更新`convert_to_iob2.py`中的标签映射
3. 重新训练模型

### 自定义模型

可以替换默认的mBERT模型为其他预训练模型：

```python
# 在train.py中修改
MODEL_NAME = "bert-base-chinese"  # 中文BERT
# 或者
MODEL_NAME = "hfl/chinese-bert-wwm"  # 中文全词覆盖BERT
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 数据贡献

如果你有标注好的漫画文件名数据，欢迎分享以改进模型性能。

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情。

## 🙏 致谢

- Hugging Face团队提供的Transformers库
- Google Research的BERT模型

## 📞 联系

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 本项目专注于教育和技术研究目的，请尊重版权，仅使用合法获取的数据进行训练。