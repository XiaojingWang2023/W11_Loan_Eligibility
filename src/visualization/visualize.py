
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=16)
    plt.savefig("confusion_matrix.png")
    plt.close()

def plot_probability_distribution(probabilities, threshold=0.7):
    """
    Plot the probability distribution of predictions with a threshold marker.
    
    Args:
        probabilities (numpy.ndarray): Array of predicted probabilities for the positive class.
        threshold (float, optional): Decision threshold for classification. Default is 0.5.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(probabilities, bins=20, kde=True)
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Probability Distribution of Predictions', fontsize=16)
    plt.savefig("probability_distribution.png")
    plt.close()