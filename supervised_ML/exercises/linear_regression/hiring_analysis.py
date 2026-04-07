import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.core.linear_regression import LinearRegression
from utils.data_preprocessing import fix_missing_values, words_to_numbers, clean_data

def run_hiring_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hiring.csv")
    df = clean_data(pd.read_csv(data_path))
    
    # Preprocess
    df['experience'] = df['experience'].fillna('zero')
    df = words_to_numbers(df, ['experience'])
    df = fix_missing_values(df, strategy='median')
    
    features = ['experience', 'test_score', 'interview_score']
    X, y = df[features].values, df['salary'].values
    
    # 1. Custom Model
    model = LinearRegression()
    model.fit(X, y)
    
    candidates = [[2, 9, 6], [12, 10, 10]]
    print("--- Hiring Salary Analysis (Multivariate) ---")
    for cand in candidates:
        pred = model.predict([cand]).item()
        print(f"Candidate {cand} -> Estimated Salary: ${pred:,.2f}")
    
    # 2. Validation with Sklearn
    sk_model = SklearnLR().fit(X, y)
    if np.allclose(model.predict(X), sk_model.predict(X)):
        print("\n✅ Success! Custom Model matches Sklearn results.")
    
    # 3. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['blue', 'green', 'orange']
    
    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], y, color='red', marker='+', label='Actual')
        # Create a trend line for each feature individually for visualization
        temp_model = SklearnLR().fit(df[[feature]].values, y)
        axes[i].plot(df[feature], temp_model.predict(df[[feature]].values), color=colors[i], label='Trend')
        axes[i].set_xlabel(feature.capitalize())
        axes[i].set_ylabel("Salary ($)")
        axes[i].set_title(f"{feature.capitalize()} vs Salary")
        axes[i].legend()
        axes[i].grid(True)

    plt.suptitle("Impact of Experience, Test Scores, and Interview Scores on Salary")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plots_dir = os.path.join(current_dir, "../../plots/linear_regression")
    os.makedirs(plots_dir, exist_ok=True)
    plot_file = os.path.join(plots_dir, "hiring_analysis.png")
    plt.savefig(plot_file)
    print(f"\nVisualization saved to: {os.path.abspath(plot_file)}")

if __name__ == "__main__":
    run_hiring_exercise()
