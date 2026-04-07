import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.core.classification.random_forest import RandomForest
from utils.data_preprocessing import clean_data

def run_retention_rf_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hr.csv")
    df = clean_data(pd.read_csv(data_path))
    
    # 1. Feature Engineering
    # We'll use more features this time for the Forest to learn complex patterns
    # Categorical: salary, Department
    salary_dummies = pd.get_dummies(df.salary, prefix="salary")
    dept_dummies = pd.get_dummies(df.Department, prefix="dept")
    
    # Numerical features
    num_features = ['satisfaction_level', 'last_evaluation', 'number_project', 
                    'average_montly_hours', 'time_spend_company', 'Work_accident', 
                    'promotion_last_5years']
    
    X = pd.concat([df[num_features], salary_dummies, dept_dummies], axis=1).values
    y = df.left.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # 2. Custom Random Forest
    print("--- HR Employee Retention: Random Forest Analysis ---")
    print(f"Dataset: {len(X)} records, {X.shape[1]} features.")
    print("Training Custom Forest (10 trees, max_depth 15)... This may take a few seconds.")
    
    model = RandomForest(n_trees=10, max_depth=15)
    model.fit(X_train, y_train)
    
    # Accuracy check
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print(f"Custom Random Forest Accuracy: {accuracy:.4f}")
    
    # 3. Sklearn Baseline for parity
    sk_model = SklearnRF(n_estimators=10, max_depth=15, random_state=42)
    sk_model.fit(X_train, y_train)
    sk_acc = sk_model.score(X_test, y_test)
    print(f"Sklearn Random Forest Accuracy: {sk_acc:.4f}")
    
    # 4. Visualization: Confusion Matrix comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Custom CM
    cm_custom = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Greens', ax=ax1)
    ax1.set_title(f"Custom Random Forest (Acc: {accuracy:.4f})")
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Sklearn CM
    cm_sk = confusion_matrix(y_test, sk_model.predict(X_test))
    sns.heatmap(cm_sk, annot=True, fmt='d', cmap='Purples', ax=ax2)
    ax2.set_title(f"Sklearn Random Forest (Acc: {sk_acc:.4f})")
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')

    plt.suptitle("Confusion Matrix Comparison: Employee Retention Prediction")
    
    plots_dir = os.path.join(current_dir, "../../plots/random_forest")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "hr_retention_matrix.png"))
    print(f"\nAnalysis Plot saved to: {os.path.abspath(plots_dir)}")

if __name__ == "__main__":
    run_retention_rf_exercise()
