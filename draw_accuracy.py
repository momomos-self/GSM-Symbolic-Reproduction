import json
import matplotlib.pyplot as plt
import seaborn as sns

# 文件路径
file_path = "accuracy_GPT4mini_no_finetune_8shot_50_groups.jsonl"  # 替换为你的文件路径

# 加载数据
accuracies = []

# 逐行读取 JSONL 文件
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())
        accuracies.append(record["accuracy"])

# 设置基准值
# baseline = 0.66
baseline_30 = 0.87

# 设置画图风格
sns.set_theme(style="ticks")

# 颜色：淡蓝色、紫色和深蓝色
color = "#5DADE2"  # 选择一个颜色（例如淡蓝色）

# 绘制直方图和分布曲线
plt.figure(figsize=(8, 6))
sns.histplot(accuracies, kde=True, bins=10, color=color,
             edgecolor="#5DADE2", linewidth=1.5)

# 添加平均值参考线
mean_accuracy = sum(accuracies) / len(accuracies)

# 添加基准线
# plt.axvline(x=baseline, color="black", linestyle="--", label=f"GSM8K(100) {baseline:.1%}")
plt.axvline(x=baseline_30, color="gray", linestyle="--", label=f"GSM8K(30) {baseline_30:.1%}")
plt.axvline(x=mean_accuracy, color=color, linestyle="--", label=f"Both {mean_accuracy:.1%}")

# 添加标题和坐标轴标签
# GPT-3.5-turbo-0125 GPT4mini
plt.title("GPT4mini", fontsize=14)
plt.xlabel("GSM Symbolic Accuracy (%) - (8s CoT)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

# 添加图例
plt.legend()

# 保存图表
output_file = "accuracy_distribution_GPT4mini_30.png"  # 设置保存文件路径
plt.savefig(output_file, dpi=300, bbox_inches="tight")  # 高分辨率保存

# 显示图表
plt.show()
