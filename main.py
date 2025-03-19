# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Swapnil Kangralkar',
#     license='',
# )

from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_confusion_matrix, plot_probability_distribution
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_LRmodel
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/credit.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    X, y = create_dummy_vars(df)

    # Train the logistic regression model
    model, X_test_scaled, y_test = train_LRmodel(X, y)

    # Evaluate the model
    accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:")
    print(confusion_mat)
    
    # Visualizations
    plot_confusion_matrix(confusion_mat, title="Logistic Regression Confusion Matrix")
    
    # Get probability predictions for visualization
    probabilities = model.predict_proba(X_test_scaled)[:,1]
    plot_probability_distribution(probabilities)
