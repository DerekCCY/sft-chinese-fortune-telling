import numpy as np
import matplotlib.pyplot as plt

# === Data ===
models = ["Gemini-2.5-flash", "Qwen3-4B", "Ziwei", "Ziwei-RAG"]
structural = np.array([0.16, 0.10, 0.17, 0.09])
interpretation = np.array([0.45, 0.45, 0.45, 0.46])
text_similarity = np.array([0.79, 0.82, 0.81, 0.83])

# Group data to find max values easily
metrics_data = [structural, interpretation, text_similarity]
metrics_names = ['Structural', 'Interpretation', 'Text Similarity']

# Setup plot
x = np.arange(len(metrics_names))
width = 0.15 

plt.figure(figsize=(10, 6))

# Plot bars (Loop by models to keep colors consistent)
for i, model_name in enumerate(models):
    scores = [structural[i], interpretation[i], text_similarity[i]]
    position = x + (i - 1.5) * width
    plt.bar(position, scores, width, label=model_name)

# === Add Best Performance Dashed Lines ===
for i, metric_scores in enumerate(metrics_data):
    # Find max score for this metric
    max_score = np.nanmax(metric_scores)
    
    # Calculate the span of the line (covering all 4 bars for this metric)
    # 4 bars * width 0.15 = 0.6 total width. From center, it extends +/- 0.3 approx
    # Precise calculation: Leftmost bar center - 0.5w, Rightmost bar center + 0.5w
    line_start = x[i] - 2 * width  # (1.5 + 0.5) * width
    line_end = x[i] + 2 * width
    
    plt.plot([line_start, line_end], [max_score, max_score], 
            color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

# Styling
plt.xticks(x, metrics_names, fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Evaluation Metrics by Model', fontsize=14)
plt.legend(title="Models")
plt.tight_layout()
plt.savefig('visualization/bar_compare.jpg')