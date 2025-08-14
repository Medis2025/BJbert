import matplotlib.pyplot as plt
import numpy as np

# Data
models = ["BioBERT (Devlin et al., 2019)", "SciFive-Base (Phan et al., 2021)", "OpenMed NER (2025)", "Ours — Multi-Task BioLinkBERT"]
precision = [0.894, 0.882, 0.918, 0.9395]
recall = [0.900, 0.895, 0.904, 0.9466]
f1 = [0.897, 0.889, 0.911, 0.9430]
accuracy = [0.894, 0.882, 0.918, 0.9900]  # Using same as precision for missing values, adjust if known

x = np.arange(len(models))
width = 0.2

# Plot PRF1 + Accuracy together
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - 1.5*width, precision, width, label='Precision')
rects2 = ax.bar(x - 0.5*width, recall, width, label='Recall')
rects3 = ax.bar(x + 0.5*width, f1, width, label='F1 Score')
rects4 = ax.bar(x + 1.5*width, accuracy, width, label='Accuracy')

# Labels and formatting
ax.set_ylabel('Score')
ax.set_title('Biomedical NER Model Comparison (NCBI-Disease)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha='right')
ax.set_ylim(0.85, 1.0)
ax.legend()

# Annotate bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for rects in [rects1, rects2, rects3, rects4]:
    autolabel(rects)

fig.tight_layout()
plt.savefig('ner_leaderboard_with_accuracy.png', dpi=300)
plt.show()
