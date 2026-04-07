import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR

# Add root directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.core.logistic_regression import LogisticRegression
from utils.data_preprocessing import clean_data

def run_hr_analysis():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hr.csv")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    print("--- HR Employee Retention EDA ---")
    print(f"Data Shape: {df.shape}")
    
    # Analysis 1: Group by 'left' to see trends
    retention_stats = df.groupby('left').mean(numeric_only=True)
    print("\nMean stats for employees who stayed (0) vs left (1):")
    print(retention_stats)
    
    # 2. Exploratory Data Analysis (EDA) - Visualizations
    plots_dir = os.path.join(current_dir, "../../plots/logistic_regression")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Impact of Salary on Retention
    pd.crosstab(df.salary, df.left).plot(kind='bar', figsize=(10, 6))
    plt.title("Impact of Salary on Retention")
    plt.xlabel("Salary Level")
    plt.ylabel("Number of Employees")
    plt.savefig(os.path.join(plots_dir, "salary_impact.png"))
    
    # Plot 2: Correlation between Department and Retention
    pd.crosstab(df.Department, df.left).plot(kind='bar', figsize=(12, 6))
    plt.title("Employee Retention per Department")
    plt.xlabel("Department")
    plt.ylabel("Number of Employees")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(plots_dir, "department_impact.png"))
    
    print(f"\nPlots saved to: {plots_dir}")
    
    # 3. Feature Selection
    subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
    
    # Handle categorical salary (One-Hot Encoding)
    salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
    df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns').drop('salary', axis='columns')
    
    X = df_with_dummies
    y = df.left
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # 4. Custom Logistic Regression Model
    model = LogisticRegression(learning_rate=0.001, iterations=10000)
    # Scaling X for gradient descent stability
    X_train_scaled = X_train / X_train.max()
    X_test_scaled = X_test / X_train.max()
    
    model.fit(X_train_scaled, y_train)
    
    # Measurement
    def get_accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    custom_pred = model.predict(X_test_scaled)
    custom_acc = get_accuracy(y_test, custom_pred)
    
    print("\n--- Model Performance ---")
    print(f"Custom Model Accuracy: {custom_acc:.4f}")
    
    # 5. Sklearn Verification
    sk_model = SklearnLR(max_iter=1000)
    sk_model.fit(X_train, y_train)
    sk_acc = sk_model.score(X_test, y_test)
    print(f"Sklearn Model Accuracy: {sk_acc:.4f}")
    
    if np.isclose(custom_acc, sk_acc, atol=0.05):
        print("\n✅ Success! Custom implementation performance matches Sklearn baseline.")

if __name__ == "__main__":
    run_hr_analysis()
