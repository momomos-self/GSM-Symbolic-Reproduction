import json
import os
from tqdm import tqdm

data_path = "data/test.jsonl"
message_path = "data/test_messages.jsonl"

# 如果 message_path 已存在，清空文件内容，避免重复追加
if os.path.exists(message_path):
    open(message_path, "w").close()

# 统计数据总量，用于初始化 tqdm
with open(data_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# 加载数据集
with open(message_path, "a", encoding="utf-8") as file_json:
    with open(data_path, 'r', encoding='utf-8') as f, tqdm(total=total_lines, desc="Processing", unit="line") as pbar:
        for line in f:
            record = json.loads(line)
            data_message = {"messages":[{"role":"system", "content":"As an expert problem solver, solve step by step the following mathematical questions."},
                               {"role":"user", "content":record["question"]},
                               {"role":"assistant", "content":record["answer"]}]}  # +"<|endoftext|>"?
            file_json.write(json.dumps(data_message, ensure_ascii=False))
            file_json.write("\n")
            # 更新进度条
            pbar.update(1)

