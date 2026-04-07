import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from gradient_descent import GradientDescent
from utils.data_preprocessing import clean_data

def run_test_scores_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/test_scores.csv")
    df = clean_data(pd.read_csv(data_path, sep=r'\s+'))
    
    X, y = df['math'].values, df['cs'].values
    
    # 1. Custom Gradient Descent
    model = GradientDescent(learning_rate=0.0002, iterations=100000)
    model.fit(X, y)
    
    print("--- Test Scores Gradient Descent Analysis ---")
    print(f"GD Results -> m: {model.m:.4f}, b: {model.b:.4f}")
    
    # 2. Validation with Sklearn
    sk_model = SklearnLR().fit(X.reshape(-1, 1), y)
    print(f"Sklearn Results -> m: {sk_model.coef_[0]:.4f}, b: {sk_model.intercept_:.4f}")
    
    if np.isclose(model.m, sk_model.coef_[0], rtol=1e-2):
        print("\n✅ Success! Custom Gradient Descent converged to Sklearn baseline.")

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='red', marker='+', label='Actual Scores')
    plt.plot(X, model.predict(X), color='blue', label='Gradient Descent Line')
    plt.plot(X, sk_model.predict(X.reshape(-1, 1)), color='green', linestyle='--', label='Sklearn Line')
    plt.xlabel("Math Score"); plt.ylabel("CS Score"); plt.title("Math vs CS Scores (Gradient Descent vs Sklearn)")
    plt.legend(); plt.grid(True)
    
    plt.savefig(os.path.join(current_dir, "example_test_scores_analysis.png"))
    print(f"\nVisualization saved to: {current_dir}")

if __name__ == "__main__":
    run_test_scores_exercise()
