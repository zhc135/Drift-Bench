import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# --- 数据准备 ---
data = {
    'Model': ['Claude Sonnet 4.5', 'DeepSeek-V3.1', 'Gemini 2.5 Flash', 'GPT-4o', 'GPT-OSS-20B', 'Phi-4', 'Gemma-3-12B-IT', 'Llama-3.1-8B', 'Qwen-2.5-7B'],
    'Info': [76.14, 35.27, 49.88, 16.19, 20.17, 21.56, 44.24, 16.61, 12.67],
    'Logic': [75.44, 69.42, 65.07, 70.38, 62.13, 67.21, 46.44, 45.17, 54.34],
    'Lang': [73.54, 75.35, 71.53, 77.74, 77.83, 73.46, 68.46, 67.28, 79.74],
    'Total': [75.03, 56.93, 61.46, 44.58, 46.03, 47.39, 52.01, 36.96, 38.01],
    'BERTScore': [54.82, 52.11, 52.37, 51.24, 50.68, 51.26, 51.96, 50.26, 49.62],
    'ROUGE-2': [81.46, 76.94, 79.08, 73.50, 71.90, 75.60, 78.57, 76.90, 71.43]
}

df = pd.DataFrame(data)
auto_metrics = ['BERTScore', 'ROUGE-2']
other_metrics = ['Info', 'Logic', 'Lang', 'Total']

# --- 画布设置 ---
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 2]})
colors = sns.color_palette("Spectral", n_colors=len(df))
bar_width = 0.07

# --- 1. 左侧：自动指标（柱状图透明度调为 0.4） ---
x_auto = np.arange(len(auto_metrics))
for i, model in enumerate(df['Model']):
    pos = x_auto + i*bar_width - (len(df)*bar_width)/2
    vals = df.iloc[i][auto_metrics].values
    axes[0].bar(pos, vals, bar_width, label=model, color=colors[i], alpha=0.4, edgecolor='none')  # 原 alpha=0.8 → 0.4

# 为每个指标画出差异框
for j, metric in enumerate(auto_metrics):
    m_min, m_max = df[metric].min(), df[metric].max()
    diff = m_max - m_min
    axes[0].add_patch(plt.Rectangle((j-0.4, m_min), 0.8, diff, color='red', alpha=0.1, zorder=0))
    axes[0].annotate(f'Gap: {diff:.1f}', xy=(j, m_max), xytext=(0, 15),
                     textcoords='offset points', ha='center', color='red',
                     fontweight='bold', fontsize=20,
                     arrowprops=dict(arrowstyle='->', color='red'))

axes[0].set_title("Automated Metrics (Saturated)", fontsize=26, pad=30, fontweight='bold', color='#666666')

# --- 2. 右侧：深度指标（柱状图透明度调为 0.8） ---
x_other = np.arange(len(other_metrics))

# 先绘制柱状图
for i, model in enumerate(df['Model']):
    vals = df.iloc[i][other_metrics].values
    pos = x_other + i*bar_width - (len(df)*bar_width)/2
    axes[1].bar(pos, vals, bar_width, color=colors[i], alpha=0.8, edgecolor='none')  # 原 alpha=0.4 → 0.8

# 在每个单维度内部，连接不同模型在该维度上的得分（按模型顺序）
for j, metric in enumerate(other_metrics):
    x_points = []
    y_points = []
    for i, model in enumerate(df['Model']):
        x_pos = x_other[j] + i*bar_width - (len(df)*bar_width)/2
        y_val = df.iloc[i][metric]
        x_points.append(x_pos)
        y_points.append(y_val)
    axes[1].plot(x_points, y_points, marker='o', markersize=5, linewidth=2, alpha=0.9,
                 color='gray', linestyle='--', label='_nolegend_')

axes[1].set_title("Our Evaluation (High Discriminative)", fontsize=28, pad=30, fontweight='bold')

# --- 3. 细节美化 ---
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.tick_params(axis='y', labelsize=20)

axes[0].set_xticks(x_auto)
axes[0].set_xticklabels(auto_metrics, fontsize=24, fontweight='bold')
axes[1].set_xticks(x_other)
axes[1].set_xticklabels(other_metrics, fontsize=24, fontweight='bold')

# 底部图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.12),
           frameon=True, facecolor='white', edgecolor='#eeeeee', fontsize=22)

plt.tight_layout()

# --- 保存图片到当前目录（PDF 格式） ---
plt.savefig('comparison_plot.pdf', dpi=300, bbox_inches='tight')
print("图片已保存为 comparison_plot.pdf")

plt.show()