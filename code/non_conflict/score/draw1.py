import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 配置部分
# ==========================================
# 设置中文字体，防止乱码 (根据操作系统选择合适的字体)
# Windows: SimHei, Mac: Arial Unicode MS or Heiti TC, Linux: WenQuanYi Micro Hei
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 确保输出目录存在 (当前文件夹)
output_dir = "."
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 模型名称列表 (保持顺序一致)
models = [
    'deepseek-v3.1', 'gemma-3-12b-it', 'gpt-4o', 'gpt-4o-mini', 
    'gpt-oss-20b', 'llama-3.1-8B-instruct', 'phi-4', 'qwen-2.5-7b-instruct'
]
# 用于 x 轴显示的简短名称
model_short_names = [m.split('-')[0] if m != 'llama-3.1-8B-instruct' else 'llama-3.1' for m in models]

# 维度名称
dimensions = ['Info', 'Logic', 'Language']

# ==========================================
# 数据准备 (严格对照 detailed_summary_report.md)
# ==========================================

# 1. 评分数据 (S_info, S_logic, S_lang, S_total)
# 顺序：info, logic, lang, total
scores_en = np.array([
    [24.88, 49.54, 56.78, 41.21],  # deepseek
    [35.03, 22.17, 46.87, 33.14],  # gemma
    [11.08, 49.54, 60.43, 32.13],  # gpt-4o
    [11.66, 50.93, 60.01, 32.91],  # gpt-4o-mini
    [11.19, 41.56, 60.57, 30.43],  # gpt-oss-20b
    [9.47,  22.58, 45.27, 21.31],  # llama
    [13.27, 46.87, 54.09, 32.28],  # phi-4
    [10.01, 32.38, 64.02, 27.48],  # qwen
])

scores_cn = np.array([
    [40.61, 57.43, 46.01, 47.52],  # deepseek
    [38.42, 44.75, 44.34, 42.40],  # gemma
    [26.06, 59.05, 58.37, 44.78],  # gpt-4o
    [12.79, 57.43, 50.70, 33.39],  # gpt-4o-mini
    [25.82, 47.30, 48.41, 38.96],  # gpt-oss-20b
    [26.18, 27.68, 47.63, 32.56],  # llama
    [32.99, 31.21, 45.80, 36.13],  # phi-4
    [29.52, 45.17, 48.52, 40.14],  # qwen
])

# 2. 问题数量汇总 (Logic, Lang, Info, Total)
errors_en = np.array([
    [76, 245, 301, 622],   # deepseek
    [163, 328, 227, 718],  # gemma
    [76, 218, 476, 770],   # gpt-4o
    [73, 221, 465, 759],   # gpt-4o-mini
    [95, 217, 474, 786],   # gpt-oss-20b
    [161, 343, 510, 1014], # llama
    [82, 266, 437, 785],   # phi-4
    [122, 193, 498, 813],  # qwen
])

errors_cn = np.array([
    [60, 336, 195, 591],   # deepseek
    [87, 352, 207, 646],   # gemma
    [57, 233, 291, 581],   # gpt-4o
    [60, 294, 445, 799],   # gpt-4o-mini
    [81, 314, 293, 688],   # gpt-oss-20b
    [139, 321, 290, 750],  # llama
    [126, 338, 240, 704],  # phi-4
    [86, 313, 264, 663],   # qwen
])

# 3. 信息完整性问题小类 (Global, Paragraph) - 英文
info_sub_en = np.array([
    [123, 178], # deepseek
    [120, 107], # gemma
    [232, 244], # gpt-4o
    [222, 243], # gpt-4o-mini
    [216, 258], # gpt-oss-20b
    [237, 273], # llama
    [195, 242], # phi-4
    [288, 210], # qwen
])

# 4. 逻辑问题小类 (Time, Coherence, Causality, Other) - 英文
logic_sub_en = np.array([
    [26, 36, 10, 4],  # deepseek
    [45, 94, 17, 7],  # gemma
    [31, 25, 16, 4],  # gpt-4o
    [22, 29, 17, 5],  # gpt-4o-mini
    [30, 46, 13, 6],  # gpt-oss-20b
    [40, 81, 28, 12], # llama
    [31, 29, 19, 3],  # phi-4
    [30, 54, 32, 6],  # qwen
])

# ==========================================
# 绘图方案 1: 多维得分对比雷达图 (Top 4 模型)
# ==========================================
def plot_radar_single(scores, language, top_k=4):
    """绘制单个语言的雷达图"""
    # 按总分排序选择前 K 个模型
    totals = scores[:, 3]
    top_idx = np.argsort(totals)[::-1][:top_k]
    
    angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist() + [0]
    dimensions_closed = dimensions + [dimensions[0]]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']
    
    for idx, (i, color) in enumerate(zip(top_idx, colors)):
        values = scores[i][:3].tolist() + [scores[i][0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=models[i], color=color, markersize=6)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, size=12, weight='bold')
    ax.set_ylim(0, 70)
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(['20', '40', '60'], color="grey", size=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'Top-{top_k} Model Profiles ({language} News)', size=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=False)
    
    filename = os.path.join(output_dir, f'radar_{language.lower()}_top{top_k}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"Saved: {filename}")

# 生成英文和中文雷达图 (分开保存)
plot_radar_single(scores_en, 'EN')
plot_radar_single(scores_cn, 'CN')

# ==========================================
# 绘图方案 2: 维度得分分组柱状图 + 问题数量叠加
# ==========================================
def plot_bar_with_errors(scores, errors, language):
    """绘制单个语言的柱状图 + 错误数折线"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(models))
    width = 0.25
    
    # 绘制三个维度的得分
    rects1 = ax.bar(x_pos - width, scores[:, 0], width, label='Info', color='#E41A1C')
    rects2 = ax.bar(x_pos, scores[:, 1], width, label='Logic', color='#377EB8')
    rects3 = ax.bar(x_pos + width, scores[:, 2], width, label='Language', color='#4DAF4A')
    
    # 添加总问题数量折线 (右轴)
    ax2 = ax.twinx()
    total_errors = errors[:, 3]
    ax2.plot(x_pos, total_errors, 'ko--', linewidth=2, markersize=8, label='Total Errors')
    ax2.set_ylabel('Total Error Count', color='black', weight='bold', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, max(total_errors) * 1.2)
    
    # 设置标签和标题
    ax.set_title(f'{language} News: Dimension Scores vs. Error Burden', weight='bold', pad=15, fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Diagnostic Score', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_short_names, rotation=45, ha='right')
    ax.set_ylim(0, 70)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper left')
    
    filename = os.path.join(output_dir, f'dimension_scores_{language.lower()}.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"Saved: {filename}")

# 生成英文和中文柱状图 (分开保存)
plot_bar_with_errors(scores_en, errors_en, 'EN')
plot_bar_with_errors(scores_cn, errors_cn, 'CN')

# ==========================================
# 绘图方案 3: 问题类型构成堆叠图 (拆分信息 vs 逻辑)
# ==========================================
def plot_stacked_info():
    """绘制信息完整性问题堆叠图 (英文)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(models))
    width = 0.6
    
    global_errors = info_sub_en[:, 0]
    paragraph_errors = info_sub_en[:, 1]
    
    ax.bar(x_pos, global_errors, width, label='Global Omission', color='#E41A1C')
    ax.bar(x_pos, paragraph_errors, width, bottom=global_errors, label='Paragraph Omission', color='#FF7F00')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_short_names, rotation=45, ha='right')
    ax.set_ylabel('Error Count', fontsize=12)
    ax.set_title('Information Completeness Errors (EN News)', weight='bold', pad=15, fontsize=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    filename = os.path.join(output_dir, 'error_composition_info_en.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"Saved: {filename}")

def plot_stacked_logic():
    """绘制逻辑连贯性问题堆叠图 (英文)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(models))
    width = 0.6
    
    bottom = np.zeros(len(models))
    categories = ['Time', 'Coherence', 'Causality', 'Other']
    colors = ['#377EB8', '#4DAF4A', '#984EA3', '#FFFF33']
    
    for i, (cat, color) in enumerate(zip(categories, colors)):
        values = logic_sub_en[:, i]
        ax.bar(x_pos, values, width, bottom=bottom, label=cat, color=color)
        bottom += values
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_short_names, rotation=45, ha='right')
    ax.set_ylabel('Error Count', fontsize=12)
    ax.set_title('Logical Coherence Errors (EN News)', weight='bold', pad=15, fontsize=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    filename = os.path.join(output_dir, 'error_composition_logic_en.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"Saved: {filename}")

# 生成堆叠图 (分开保存)
plot_stacked_info()
plot_stacked_logic()

# ==========================================
# 绘图方案 4: 得分 - 错误数量散点图 (验证评分公式)
# ==========================================
def plot_scatter_validation():
    """绘制信息维度得分 vs 错误数量散点图 (英文)"""
    from scipy import stats
    
    info_scores = scores_en[:, 0]  # Info 维度得分
    info_errors = errors_en[:, 2]  # 信息问题总数
    
    # 计算理论得分曲线
    E_avg = np.array(info_errors) / 150
    lambda_info = np.log(2)  # 0.6931
    theoretical_scores = 100 * np.exp(-lambda_info * E_avg)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 散点：实际得分
    scatter_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', 
                      '#FF7F00', '#FFFF33', '#A65628', '#F781BF']
    
    ax.scatter(info_errors, info_scores, c=scatter_colors, s=120, 
               edgecolors='black', linewidth=1.5, alpha=0.8, label='Observed Score')
    
    # 曲线：理论得分
    x_curve = np.linspace(min(info_errors), max(info_errors), 100)
    y_curve = 100 * np.exp(-lambda_info * (x_curve / 150))
    ax.plot(x_curve, y_curve, 'k--', linewidth=2, label='Theoretical Decay ($S=100e^{-λE}$)')
    
    # 标注模型名称
    for i, model in enumerate(models):
        ax.annotate(model_short_names[i], 
                    (info_errors[i], info_scores[i]), 
                    textcoords="offset points", xytext=(8, -8), ha='left', fontsize=9)
    
    ax.set_xlabel('Information Error Count (per 150 articles)', weight='bold', fontsize=12)
    ax.set_ylabel('Info Dimension Score', weight='bold', fontsize=12)
    ax.set_title('Validation of Exponential Decay Scoring Mechanism (EN)', weight='bold', pad=15, fontsize=14)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # 添加相关系数
    corr, p_val = stats.pearsonr(info_errors, info_scores)
    ax.text(0.02, 0.98, f'Pearson r = {corr:.3f}\np = {p_val:.3e}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    filename = os.path.join(output_dir, 'scoring_validation_scatter_en.png')
    plt.savefig(filename, bbox_inches='tight', dpi=300, format='png')
    plt.close()
    print(f"Saved: {filename}")

# 生成散点图
plot_scatter_validation()

print("\n所有图片已生成完毕，请检查当前文件夹。")