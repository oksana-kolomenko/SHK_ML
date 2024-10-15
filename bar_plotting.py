import numpy as np
import matplotlib.pyplot as plt

def plot_bar_chart(labels, train_scores, test_score_medians, test_score_mins, test_score_maxs,
                   ylabel='AUC'):
    """
    Saves a bar chart comparing train scores and test scores with error bars.

    Parameters:
    - labels (list of str): Labels for each group (e.g., model types).
    - train_scores (array of float): Training scores for each group.
    - test_score_medians (array of float): Median test scores for each group.
    - test_score_mins (array of float): Minimum test scores for each group.
    - test_score_maxs (array of float): Maximum test scores for each group.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None: Saves the bar chart.
    """
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, train_scores, width, label='Train AUC')
    rects2 = ax.bar(x + width / 2, test_score_medians, width, yerr=np.stack(
        [test_score_medians - test_score_mins, test_score_maxs - test_score_medians]),
        label='Test AUC', capsize=5)
    
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("bar_chart.pdf")
