import json
import argparse
import random
import os

def convert_chunks_to_iob2(input_path, train_path, valid_path, test_path, train_ratio=0.8, valid_ratio=0.1):
    """
    将“文本块”JSONL 格式数据转换为 IOB2 格式，并分割成训练集、验证集和测试集。
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 -> {input_path}")
        return

    print(f"成功读取 {len(lines)} 条“文本块”标注数据。")

    random.shuffle(lines)
    
    total_lines = len(lines)
    train_split_point = int(total_lines * train_ratio)
    valid_split_point = train_split_point + int(total_lines * valid_ratio)

    train_lines = lines[:train_split_point]
    valid_lines = lines[train_split_point:valid_split_point]
    test_lines = lines[valid_split_point:]

    print(f"将分割为 {len(train_lines)} 条训练数据, {len(valid_lines)} 条验证数据和 {len(test_lines)} 条测试数据。")

    for dataset_lines, output_path in [
        (train_lines, train_path),
        (valid_lines, valid_path),
        (test_lines, test_path)
    ]:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in dataset_lines:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    spans = data.get('spans', [])
                    
                    if not spans:
                        continue

                    # 根据 spans 生成完整的 IOB2 序列
                    for span in spans:
                        text = span.get('text', '')
                        label = span.get('label', 'O')
                        
                        if not text:
                            continue

                        # 如果标签不是 'O'，应用 B- 和 I- 规则
                        if label != 'O':
                            # 为这个文本块的第一个字符应用 B- 标签
                            f_out.write(f"{text[0]} B-{label}\n")
                            # 为剩余的字符应用 I- 标签
                            for char in text[1:]:
                                f_out.write(f"{char} I-{label}\n")
                        else:
                            # 如果标签是 'O'，所有字符都是 'O'
                            for char in text:
                                f_out.write(f"{char} O\n")

                    # 在每个样本（即每行 JSON）处理完毕后，写入一个空行作为分隔符
                    f_out.write("\n")

                except json.JSONDecodeError:
                    print(f"警告: 无效的 JSON 行，已跳过: {line.strip()}")
                    continue
    
    print(f"转换完成！\n训练数据已保存至: {train_path}\n验证数据已保存至: {valid_path}\n测试数据已保存至: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将“文本块”JSONL 标注数据转换为 IOB2 格式。")
    parser.add_argument("--input_file", type=str, default="./data/chunked_annotations.jsonl", help="输入的“文本块”JSONL 文件路径")
    parser.add_argument("--train_file", type=str, default="./data/train.txt", help="输出的训练集文件路径 (IOB2 格式)")
    parser.add_argument("--valid_file", type=str, default="./data/validation.txt", help="输出的验证集文件路径 (IOB2 格式)")
    parser.add_argument("--test_file", type=str, default="./data/test.txt", help="输出的测试集文件路径 (IOB2 格式)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集所占的比例")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="验证集所占的比例")
    
    args = parser.parse_args()
    convert_chunks_to_iob2(args.input_file, args.train_file, args.valid_file, args.test_file, args.train_ratio, args.valid_ratio)