import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso as SklearnLasso

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.Regression.Lasso.lasso_regression import LassoRegression

def run_lasso_analysis():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hiring.csv")
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Map words to numbers for experience
    word_to_num = {
        'five': 5, 'two': 2, 'seven': 7, 'three': 3, 'ten': 10, 'eleven': 11, np.nan: 0
    }
    df['experience'] = df['experience'].map(word_to_num)
    df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())
    
    X = df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']].values
    y = df['salary($)'].values
    
    # Scale data for gradient descent
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std

    # 1. Custom Model Training
    model = LassoRegression(alpha=0.1, lr=0.1, iterations=1000)
    model.fit(X_scaled, y_scaled)
    
    # 2. Sklearn Model
    sk_model = SklearnLasso(alpha=0.1)
    sk_model.fit(X_scaled, y_scaled)
    
    print("--- Hiring Salary Prediction (Lasso Regression - L1) ---")
    print(f"Custom Model Weights: {model.weights}")
    print(f"Sklearn Model Weights: {sk_model.coef_}")
    
    # Check if any weights were zeroed out (L1 characteristic)
    print(f"\nFeatures with zero weights (Custom): {np.where(np.isclose(model.weights, 0, atol=1e-2))[0]}")
    print(f"Features with zero weights (Sklearn): {np.where(np.isclose(sk_model.coef_, 0, atol=1e-2))[0]}")
    
    plt.figure(figsize=(10, 6))
    features = ['Experience', 'Test Score', 'Interview Score']
    x_axis = np.arange(len(features))
    plt.bar(x_axis - 0.2, model.weights, 0.4, label='Custom Lasso')
    plt.bar(x_axis + 0.2, sk_model.coef_, 0.4, label='Sklearn Lasso')
    plt.xticks(x_axis, features)
    plt.ylabel('Weight Magnitude (Scaled)')
    plt.title('Lasso Feature Selection: Custom vs Sklearn')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(current_dir, "example_lasso_multi_analysis.png"))
    print(f"\nVisualization saved to: {current_dir}/example_lasso_multi_analysis.png")

if __name__ == "__main__":
    run_lasso_analysis()
