import random
import json

# 生成 GSM Symbolic Template 问题
def generate_gsm_problem():
    # 定义变量的取值范围
    with open("../Template_data/template_randomlabel.json", encoding="utf-8") as f:
        raw_data = json.load(f)
        names = raw_data["female_names"]
        families = raw_data["male_families"]
    x = random.randint(5, 100)  # Building blocks count
    y = random.randint(5, 100)  # Stuffed animals count
    z = random.randint(5, 100)  # Stacking rings count
    total = random.randint(100, 500)  # Total toys count

    # 计算答案并确保符合条件 x + y + z + ans == total
    xyz_sum = x + y + z
    ans = total - xyz_sum
    if ans <= 0:  # 确保答案有效
        return generate_gsm_problem()

    # 随机选择名字和家庭关系
    name = random.choice(names)
    family = random.choice(families)

    # GSM 问题模板
    question_template = f"""
    When {name} watches her {family}, she gets out a variety of toys for him.
    The bag of building blocks has {x} blocks in it.
    The bin of stuffed animals has {y} stuffed animals inside.
    The tower of stacking rings has {z} multicolored rings on it.
    {name} recently bought a tube of bouncy balls, bringing her total number of toys she bought for her {family} up to {total}.
    How many bouncy balls came in the tube?
    """

    # GSM 解答模板
    answer_template = f"""
    Let T be the number of bouncy balls in the tube.
    After buying the tube of balls, {name} has {x} + {y} + {z} + T = {xyz_sum} + T = {total} toys for her {family}.
    Thus, T = {total} - {xyz_sum} = <<{total}-{xyz_sum}={ans}>>{ans} bouncy balls came in the tube.
    #### {ans}
    """

    return question_template.strip(), answer_template.strip()

# 批量生成 GSM 问题
def generate_gsm_problems(n=50):
    problems = []
    for _ in range(n):
        question, answer = generate_gsm_problem()
        problems.append({"question": question, "answer": answer})
    return problems

# 生成并打印 100 道问题
problems = generate_gsm_problems(100)
with open("../Template_data/gsm_problems.jsonl", "w", encoding="utf-8") as f:
    for i, item in enumerate(problems, start=1):
        message = {
            "question": item["question"],
            "answer": item["answer"],
        }
        f.write(json.dumps(message) + "\n")
        print(f"question {i}:\n{item['question']}\n")
        print(f"answer {i}:\n{item['answer']}\n")