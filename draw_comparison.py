import matplotlib.pyplot as plt
import seaborn as sns

# 数据
data = {
    "Category": ["GPT2_finetune\nGSM(FULL)", "GPT3_finetune\nGSM(FULL)", "GPT3_8shot\nGSM(100)", "GPT3_8shot\nGSM(30)", "GPT3_8shot\nGSM(20)", "GPT3_8shot\nGSM_Symbolic"],
    "Accuracy (%)": [2.0, 72.0, 66.0, 56.7, 55.0, 61.3]
}

# 设置绘图风格
sns.set_theme(style="ticks")

# 设置颜色
colors = ["#d3d3d3", "#87CEEB", "#FFCCCB", "#FFA07A", "#E6E6FA"]  # 自定义颜色

# 创建柱状图
plt.figure(figsize=(8, 6))
bars = sns.barplot(x="Category", y="Accuracy (%)", data=data, palette=colors, edgecolor="black")

# 添加数据标签
for bar in bars.patches:
    height = bar.get_height()
    bars.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)

# 添加标题和标签
plt.title("Models Accuracy", fontsize=14)
plt.xlabel("Model / Test Source", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)

# 保存图表
output_file = "accuracy_comparison.png"  # 设置保存文件路径
plt.savefig(output_file, dpi=300, bbox_inches="tight")  # 高分辨率保存

# 显示图
plt.tight_layout()
plt.show()
