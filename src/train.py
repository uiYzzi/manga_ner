import torch
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from seqeval.metrics import classification_report

# --- 1. 配置参数 ---
# 预训练模型选择，mBERT 是一个多语言模型，对中文支持良好且轻量
MODEL_NAME = "hfl/rbt3" 
DATA_PATH = "./data/"  # 存放 train.txt, validation.txt, test.txt 的文件夹
OUTPUT_DIR = "./manga_ner_model" # 训练好的模型保存路径

# --- 2. 加载数据集 ---
# load_dataset 会自动寻找文件夹下的 train/validation/test 文件
raw_datasets = load_dataset("text", data_files={
    "train": f"{DATA_PATH}train.txt",
    "validation": f"{DATA_PATH}validation.txt",
    "test": f"{DATA_PATH}test.txt"
})

# 从训练数据中获取标签列表
def get_label_list(examples):
    labels = []
    for line in examples["text"]:
        # 跳过空行和分隔符
        if line and not line.isspace() and "-DOCSTART-" not in line:
            # 每行的格式是 "字 标签"，例如 "放 B-TITLE"
            splits = line.split()
            if len(splits) > 1:
                labels.append(splits[-1])
    return {"labels": list(set(labels))}

# 获取所有标签，并创建 id 到 label 的映射
label_info = get_label_list(raw_datasets["train"])
label_list = sorted(label_info["labels"])
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for i, l in enumerate(label_list)}
num_labels = len(label_list)

print("标签列表:", label_list)
print("标签到ID映射:", label_to_id)

# --- 3. 预处理数据 ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    # 将每行的文本分割成 词 和 标签
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for line in examples["text"]:
        if not line or line.isspace() or "-DOCSTART-" in line:
            continue
        
        # 将整个文件名作为一个样本处理
        processed_splits = [s.split() for s in line.strip().split('\n')]
        words = [word_split[0] for word_split in processed_splits if len(word_split) >= 2]
        labels = [word_split[1] for word_split in processed_splits if len(word_split) >= 2]

        # 对整个句子进行分词
        # is_split_into_words=True 表示输入已经是分好词的列表
        tokenized_sentence = tokenizer(words, truncation=True, is_split_into_words=True)
        word_ids = tokenized_sentence.word_ids()

        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None: # 对于 [CLS], [SEP] 等特殊 token
                label_ids.append(-100) # -100 会在损失函数中被忽略
            elif word_idx != previous_word_idx: # 一个新词的开始
                label_ids.append(label_to_id[labels[word_idx]])
            else: # 同一个词的后续子词 (subword)
                 # 我们也用-100忽略，只标註每个词的第一个子词
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        tokenized_inputs["input_ids"].append(tokenized_sentence["input_ids"])
        tokenized_inputs["attention_mask"].append(tokenized_sentence["attention_mask"])
        tokenized_inputs["labels"].append(label_ids)

    return tokenized_inputs

# 由于我们的数据集每行是一个样本，需要修改一下处理方式
# 我们将每条文件名（一个样本）视为一个 "doc"
def process_conll_data(dataset):
    docs = []
    doc = ""
    for line in dataset['text']:
        if line.strip() == "" and doc.strip() != "":
            docs.append(doc.strip())
            doc = ""
        elif "-DOCSTART-" in line:
            if doc.strip() != "":
                docs.append(doc.strip())
            doc = ""
        else:
            doc += line + '\n'
    if doc.strip() != "":
        docs.append(doc.strip())
    return {"text": docs}

processed_datasets = raw_datasets.map(
    process_conll_data,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

tokenized_datasets = processed_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=processed_datasets["train"].column_names,
)


# --- 4. 定义模型和训练器 ---
# 数据整理器，负责将批次数据进行填充(padding)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# 加载预训练模型，并告知它我们的标签数量
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels, id2label=id_to_label, label2id=label_to_id
)

# 定义评估指标
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    report = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1-score": report["micro avg"]["f1-score"],
        "accuracy": report["micro avg"]["precision"], # In seqeval, micro-avg-precision is accuracy
    }

# 训练参数配置
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=8, # 根据你的显存调整
    per_device_eval_batch_size=8,
    num_train_epochs=5, # 训练轮数，可根据收敛情况调整
    weight_decay=0.01,
    eval_strategy="steps", # 每个 epoch 结束后进行一次评估
    save_strategy="steps",
    eval_steps=500, # 每 500 步评估一次
    save_steps=500,
    load_best_model_at_end=True,
    push_to_hub=False,
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 5. 开始训练 ---
print("开始训练...")
trainer.train()

# --- 6. 保存最好的模型 ---
trainer.save_model(f"{OUTPUT_DIR}/best_model")
print(f"训练完成！最好的模型已保存在 {OUTPUT_DIR}/best_model")

# --- 7. 在测试集上评估 ---
print("在测试集上评估...")
test_results = trainer.evaluate(tokenized_datasets["test"])
print(test_results)