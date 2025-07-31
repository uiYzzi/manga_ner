from transformers import pipeline
import torch

# --- 1. 加载你训练好的模型 ---
MODEL_PATH = "./manga_ner_model/best_model"
# 检查是否有 MPS (GPU) 可用，否则使用 CPU
device = 0 if torch.backends.mps.is_available() else -1

# 使用 pipeline 简化推理过程
# device=0 表示使用第一个 GPU (在 Mac 上就是 MPS)
ner_pipeline = pipeline("token-classification", model=MODEL_PATH, device=device, aggregation_strategy="simple")


# --- 2. 使用模型进行预测 ---
def parse_manga_title(filename):
    """使用 NER pipeline 解析漫画文件名"""
    print(f"\n正在解析: {filename}")
    try:
        entities = ner_pipeline(filename)
        if not entities:
            print("  -> 未识别到任何实体。")
            return None
        
        print("  -> 识别结果:")
        grouped_entities = {}
        for entity in entities:
            entity_group = entity['entity_group']
            word = entity['word']
            if entity_group not in grouped_entities:
                grouped_entities[entity_group] = []
            grouped_entities[entity_group].append(word)

        # 将结果整理成更易读的格式
        parsed_result = {key: "".join(value) for key, value in grouped_entities.items()}
        print(f"  -> 解析完成: {parsed_result}")
        return parsed_result

    except Exception as e:
        print(f"解析出错: {e}")
        return None

# --- 3. 交互式测试 ---
if __name__ == "__main__":
    test_filenames = [
        "減法累述 - 話017-024",
        "放學後的二人世界_第一卷",
        "[某某漢化] 將放言說女生之間不可能的女孩子、在百日之內徹底攻陷的百合故事 Ch.01 [Digital]",
        "只有神知道的世界 Vol.15",
    ]

    for name in test_filenames:
        parse_manga_title(name)

    # 进入交互模式
    print("\n--- 进入交互测试模式，输入 'exit' 退出 ---")
    while True:
        user_input = input("请输入要解析的漫画文件名: ")
        if user_input.lower() == 'exit':
            break
        parse_manga_title(user_input)