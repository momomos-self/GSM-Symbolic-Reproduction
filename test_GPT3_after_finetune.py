import openai
from tqdm import tqdm
import json
from extract_number import extract_last_number
import os

file_path = "data/test_messages.jsonl"
model_id = "ft:gpt-3.5-turbo-0125:personal::AZzPfXHb"
api_key = os.getenv('OPENAI_API_KEY')

# 模型加载
client = openai.OpenAI(
    api_key = api_key
)

# 模型推理
write_file = open("result_GPT.jsonl", "a", encoding="utf-8")
predict_right = 0  # 预测正确数
test_examples = 0  # 样例数量
skip_example = 0  # 跳过数量

# 统计数据总量，用于初始化 tqdm
with open(file_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

with open(file_path, mode='r', encoding='utf-8') as file, tqdm(total=total_lines, desc="Processing", unit="line") as pbar:
    for line in file:
        test_examples += 1
        try:
            line = line.strip()

            # 尝试解析 JSON 行
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                skip_example += 1
                test_examples -= 1
                continue  # 跳过当前行

            # 获取问题和答案
            qn = data["messages"][1]["content"]  # 问题
            answer = data["messages"][2]["content"]
            value = extract_last_number(answer)  # 答案

            # GPT 模型调用
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system",
                         "content": "As an expert problem solver, solve step by step the following mathematical questions."},
                        {"role": "user",
                         "content": qn}
                    ],
                    temperature=0.2,
                    top_p=0.9
                )
            except Exception as e:
                skip_example += 1
                test_examples -= 1
                continue  # 跳过当前问题

            text = completion.choices[0].message.content
            predict_value = extract_last_number(text)  # 预测答案

            # 处理预测值为空的情况
            if predict_value is None:
                predict_value = 0

            # 比较答案
            if abs(predict_value - value) < 0.0001:
                predict_right += 1
            write_file.write(
                json.dumps(
                    {"model_prediction": text, "predict_value": predict_value},
                    ensure_ascii=False,
                ) + "\n"
            )
            write_file.flush()

        except Exception as e:
            test_examples -= 1
            skip_example += 1
            continue  # 跳过当前行
        # 更新进度条
        pbar.update(1)


# 打印结果
print(f"Correct Predictions: {predict_right}")
print(f"Total Examples: {test_examples}")
print(f"Skip Examples: {skip_example}")
print(f"Accuracy: {predict_right / test_examples:.2f}")