import json
import argparse
import os

def validate_and_clean_jsonl(input_path):
    """
    校验 JSONL 文件中的数据，删除无效行（JSON 解析失败或 'spans' 字段为空的行）。
    """
    temp_output_path = input_path + ".tmp"
    invalid_lines_count = 0
    duplicate_lines_count = 0
    total_lines_count = 0
    processed_lines = set()

    print(f"开始校验文件: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(temp_output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            total_lines_count += 1
            try:
                data = json.loads(line)
                if not data.get('spans'):
                    invalid_lines_count += 1
                    print(f"警告: 'spans' 字段为空，已跳过: {line.strip()}")
                    continue
                
                # Check for duplicates
                line_hash = hash(line.strip())
                if line_hash in processed_lines:
                    duplicate_lines_count += 1
                    print(f"警告: 重复的行，已跳过: {line.strip()}")
                    continue
                
                processed_lines.add(line_hash)
                f_out.write(line)
            except json.JSONDecodeError:
                invalid_lines_count += 1
                print(f"警告: 无效的 JSON 行，已跳过: {line.strip()}")
                continue
    
    os.replace(temp_output_path, input_path)
    print(f"校验完成！共处理 {total_lines_count} 行，删除 {invalid_lines_count} 行无效数据，删除 {duplicate_lines_count} 行重复数据。")
    print(f"清理后的文件已保存至: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="校验并清理 JSONL 文件，删除无效行。")
    parser.add_argument("--input_file", type=str, default="./data/chunked_annotations.jsonl", help="需要校验的 JSONL 文件路径")
    
    args = parser.parse_args()
    validate_and_clean_jsonl(args.input_file)