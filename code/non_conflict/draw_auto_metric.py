import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 原始数据录入
# ==========================================
data_table1 = [
    ["en_deepseek-v3.1", 0.7734, 0.7257, 0.5873, 0.5456, 0.5615],
    ["zh_deepseek-v3.1", 0.5968, 0.4922, 0.5328, 0.3987, 0.5245],
    ["zh_claude-sonnet-4-5", 0.5913, 0.5275, 0.5667, 0.4024, 0.5394],
    ["en_gemma-3-12b-it", 0.7571, 0.7232, 0.5985, 0.5492, 0.5813],
    ["zh_gemini-2.5-flash", 0.6010, 0.5111, 0.5404, 0.4218, 0.5347],
    ["en_qwen_7b", 0.8484, 0.7702, 0.6032, 0.6031, 0.5801],
    ["zh_phi-4", 0.6200, 0.5165, 0.5587, 0.4724, 0.5812],
    ["zh_gpt-4o", 0.5898, 0.4701, 0.5189, 0.4228, 0.5409],
    ["en_phi-4", 0.7832, 0.7287, 0.5680, 0.5448, 0.5577],
    ["zh_qwen_7b", 0.6296, 0.5210, 0.5650, 0.4844, 0.5804],
    ["zh_llama3.1-8b-Instruct", 0.5764, 0.4638, 0.5164, 0.4578, 0.5590],
    ["en_gpt-oss-20b", 0.8260, 0.7313, 0.5111, 0.5212, 0.5044],
    ["en_llama3.1-8b-Instruct(1)", 0.7586, 0.7180, 0.5778, 0.5423, 0.5750],
    ["en_gpt-4o", 0.8054, 0.7335, 0.5409, 0.5393, 0.5366],
    ["en_gemini-2.5-flash", 0.7548, 0.7180, 0.5851, 0.5385, 0.5683],
    ["en_claude-sonnet-4-5", 0.7047, 0.6852, 0.6136, 0.5091, 0.5749],
    ["zh_gpt-oss-20b", 0.5464, 0.4494, 0.4769, 0.3300, 0.4355],
    ["zh_gemma-3-12b-it", 0.6106, 0.5120, 0.5472, 0.4692, 0.5641],
]

data_table2 = [
    ["en_deepseek-v3.1", 0.8676, 0.7694, 0.5532, 0.5663, 0.5211],
    ["zh_deepseek-v3.1", 0.6750, 0.6116, 0.6152, 0.3385, 0.5141],
    ["zh_claude-sonnet-4-5", 0.8192, 0.7593, 0.7674, 0.4929, 0.6186],
    ["en_gemma-3-12b-it", 0.8940, 0.7857, 0.5430, 0.5968, 0.5196],
    ["zh_gemini-2.5-flash", 0.7552, 0.6896, 0.6838, 0.4403, 0.5596],
    ["en_qwen_7b", 0.7944, 0.7143, 0.5163, 0.4564, 0.4962],
    ["zh_phi-4", 0.6576, 0.5982, 0.6188, 0.3429, 0.5315],
    ["zh_gpt-4o", 0.5574, 0.5023, 0.5264, 0.2420, 0.5031],
    ["en_phi-4", 0.8470, 0.7560, 0.5302, 0.5321, 0.5126],
    ["zh_qwen_7b", 0.6181, 0.5659, 0.5825, 0.2831, 0.5157],
    ["zh_llama3.1-8b-Instruct", 0.4993, 0.4505, 0.4716, 0.2420, 0.4950],
    ["en_gpt-oss-20b", 0.7903, 0.7190, 0.5025, 0.4578, 0.5068],
    ["en_llama3.1-8b-Instruct(1)", 0.8677, 0.7690, 0.5237, 0.5568, 0.5026],
    ["en_gpt-4o", 0.8171, 0.7350, 0.5156, 0.4935, 0.5124],
    ["en_gemini-2.5-flash", 0.8991, 0.7908, 0.5561, 0.6080, 0.5237],
    ["en_claude-sonnet-4-5", 0.9211, 0.8146, 0.5998, 0.6511, 0.5482],
    ["zh_gpt-oss-20b", 0.6362, 0.5668, 0.5598, 0.2375, 0.4086],
    ["zh_gemma-3-12b-it", 0.6872, 0.6276, 0.6334, 0.3785, 0.5498],
]

# ==========================================
# 2. 模型名称映射 (确保完整显示)
# ==========================================
name_mapping = {
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "deepseek-v3.1": "DeepSeek-V3.1",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-4o": "GPT-4o",
    "gpt-oss-20b": "GPT-OSS-20B",
    "phi-4": "Phi-4",
    "gemma-3-12b-it": "Gemma-3-12B-IT",
    "llama3.1-8b-instruct": "Llama-3.1-8B",
    "qw7b": "Qwen-2.5-7B"   # 关键修正：确保显示为 Qwen-2.5-7B
}

def process_data(raw_data, metrics_cols):
    df = pd.DataFrame(raw_data, columns=["Raw_Model"] + metrics_cols)
    # 提取语言标签
    df['Language'] = df['Raw_Model'].apply(lambda x: 'English' if x.startswith('en_') else 'Chinese')
    # 清洗模型键值：去掉语言前缀、重复标记(1)、并转为小写以匹配字典
    df['Model_Key'] = df['Raw_Model'].str.replace('en_', '').str.replace('zh_', '').str.replace(r'\(1\)', '', regex=True).str.lower()
    df['Model'] = df['Model_Key'].map(name_mapping).fillna(df['Model_Key'])
    # （可选）调试检查：打印映射后的唯一模型名
    # print("映射后的模型名称:", df['Model'].unique())
    return df

# ==========================================
# 3. 绘图函数定义
# ==========================================
def draw_performance_plot(df, metrics_list, title, save_name):
    # 转换长格式
    df_melted = df.melt(id_vars=['Model', 'Language'], value_vars=metrics_list, 
                        var_name='Metric', value_name='Score')

    # 风格设置
    sns.set_theme(style="white", font="Arial")
    plt.rcParams['axes.labelweight'] = 'bold'
    
    # 核心绘图逻辑：col="Language" 实现中英文分面
    g = sns.catplot(
        data=df_melted, kind="bar",
        x="Metric", y="Score", hue="Model",
        col="Language", palette="Set2", 
        height=5, aspect=1.3,
        edgecolor='white', linewidth=1,
        alpha=0.95
    )

    # 视觉微调
    g.set_axis_labels("", "Metric Score")
    g.set_titles("{col_name} Results", size=14, weight='bold')
    g.despine(left=True, bottom=False)
    
    # 添加网格线与轴限制
    for ax in g.axes.flat:
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_ylim(0, 1.0) 
        # 优化 X 轴标签显示
        plt.setp(ax.get_xticklabels(), size=10)

    # 精确控制图例位置：放在底部，不遮挡数据
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.02), 
                    ncol=5, title=None, frameon=False)
    
    # 调整标题与整体布局
    plt.subplots_adjust(top=0.88, bottom=0.22)
    g.fig.suptitle(title, fontsize=16, weight='bold')

    # 保存为 PDF 格式
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Success: {save_name} has been generated.")
    plt.show()

# ==========================================
# 4. 执行生成
# ==========================================

# 统一指标名称
metrics_labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "BERTScore"]

# 处理并绘制 Table 1（输出 PDF）
df1 = process_data(data_table1, metrics_labels)
draw_performance_plot(df1, metrics_labels, 
                      "Comparison with Rewritten Text (Averaged over Rewritten V1-V3)", 
                      "Comparison_with_Rewritten_Text.pdf")

# 处理并绘制 Table 2（输出 PDF）
df2 = process_data(data_table2, metrics_labels)
draw_performance_plot(df2, metrics_labels, 
                      "Comparison with Input Text", 
                      "Comparison_with_Input_Text.pdf")