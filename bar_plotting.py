import numpy as np
import matplotlib.pyplot as plt

from csv_parser import patient_summaries


def create_summaries():
    file_path = 'Summaries.txt'

    # Open the file in write mode (this will create the file if it does not exist)
    with open(file_path, 'w') as file:
        for s in patient_summaries():
            file.write(s + "\n")


def plot_bar_chart(labels, train_scores, test_means, test_stds, title='Scores by Model and Dataset', ylabel='Scores'):
    """
    Plots a bar chart comparing train scores and test scores with error bars.

    Parameters:
    - labels (list of str): Labels for each group (e.g., model types).
    - train_scores (list of float): Training scores for each group.
    - test_means (list of float): Mean test scores for each group.
    - test_stds (list of float): Standard deviations of test scores for each group.
    - title (str): Title of the chart.
    - ylabel (str): Label for the y-axis.

    Returns:
    - None: Displays the bar chart.
    """
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, train_scores, width, label='Train Scores')
    rects2 = ax.bar(x + width / 2, test_means, width, yerr=test_stds, label='Test Scores', capsize=5)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

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
    plt.show()
