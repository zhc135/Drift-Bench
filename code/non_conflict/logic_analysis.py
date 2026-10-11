import matplotlib.pyplot as plt
import numpy as np

# 数据
models = ['Claude Sonnet 4.5', 'DeepSeek-V3.1', 'Gemini 2.5 Flash', 'Gemma-3-12B', 
          'GPT-4o', 'GPT-OSS-20B', 'Llama-3.1-8B', 'Phi-4', 'Qwen-2.5-7B']
# 英文数据
en_time = [23, 28, 24, 46, 31, 33, 43, 34, 33]
en_narr = [26, 36, 51, 96, 25, 48, 89, 29, 57]
en_causal = [9, 11, 16, 17, 16, 14, 28, 20, 36]
# 中文数据
zh_time = [21, 20, 19, 25, 16, 33, 41, 36, 17]
zh_narr = [34, 32, 32, 59, 36, 38, 86, 84, 68]
zh_causal = [6, 12, 5, 19, 11, 10, 27, 20, 23]

x = np.arange(len(models))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
plt.subplots_adjust(wspace=0.3)

# 英文子图
ax1.bar(x - width/2, en_time, width, label='Time order', color='#8ECFC9', edgecolor='black')
ax1.bar(x - width/2, en_narr, width, bottom=en_time, label='Narrative incoherence', color='#FFBE7A', edgecolor='black')
ax1.bar(x - width/2, en_causal, width, bottom=np.array(en_time)+np.array(en_narr), 
        label='Causal confusion', color='#FA7F6F', edgecolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.set_ylabel('Number of logical errors', fontsize=12)
ax1.set_title('English (EN)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', frameon=True)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# 中文子图
ax2.bar(x - width/2, zh_time, width, label='Time order', color='#8ECFC9', edgecolor='black')
ax2.bar(x - width/2, zh_narr, width, bottom=zh_time, label='Narrative incoherence', color='#FFBE7A', edgecolor='black')
ax2.bar(x - width/2, zh_causal, width, bottom=np.array(zh_time)+np.array(zh_narr), 
        label='Causal confusion', color='#FA7F6F', edgecolor='black')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.set_ylabel('Number of logical errors', fontsize=12)
ax2.set_title('Chinese (ZH)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', frameon=True)
ax2.grid(axis='y', linestyle='--', alpha=0.6)

# 添加总错误数折线（右侧次要y轴）
ax2_twin = ax2.twinx()
total_en = [sum(x) for x in zip(en_time, en_narr, en_causal)]
total_zh = [sum(x) for x in zip(zh_time, zh_narr, zh_causal)]
ax2_twin.plot(x, total_en, 'o-', color='#2C3E50', linewidth=2, markersize=8, label='Total EN')
ax2_twin.plot(x, total_zh, 's--', color='#E74C3C', linewidth=2, markersize=8, label='Total ZH')
ax2_twin.set_ylabel('Total errors', fontsize=12)
ax2_twin.legend(loc='upper right', frameon=True)

plt.suptitle('Logical Error Breakdown by Model and Language', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('appendix_logic_breakdown.pdf', bbox_inches='tight')
plt.savefig('appendix_logic_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()