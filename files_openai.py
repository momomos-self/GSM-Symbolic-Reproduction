import openai
import os

api_key = os.getenv('OPENAI_API_KEY')
model_name = 'gpt-3.5-turbo-0125'
client = openai.OpenAI(
    api_key = api_key)
'''
# 1、上传文件
try:
    with open("data/train_messages.jsonl", "rb") as file:
        response = client.files.create(file=file, purpose="fine-tune")
    print("文件上传成功")
    print(client.files.list()) # 列出文件
except FileNotFoundError:
    print("找不到文件。请确保文件名和路径正确。")
except PermissionError as e:
    print(f"权限错误: {e}")
except Exception as e:
    print(f"发生意外错误: {e}")
'''

'''
# 2、微调
file_id = 'ftjob-eEsKF9by132OQxAvVeXurbPe'
print(client.files.list()) # 列出文件
client.fine_tuning.jobs.create(
    training_file = file_id,
    model = model_name
)
print(client.fine_tuning.jobs.list())  # 列出微调作业
print(client.fine_tuning.jobs.retrieve(file_id))  # 检测微调作业
'''

# 测试用
'''
completion = client.chat.completions.create(
    model = model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)
print(completion.choices[0].message)
'''