import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB as SklearnGNB
from sklearn.naive_bayes import MultinomialNB as SklearnMNB
import matplotlib.pyplot as plt
import seaborn as sns

# Path setup to root for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.Classification.NaiveBayes.naive_bayes import GaussianNB, MultinomialNB
from utils.validation import train_test_split_custom

def run_wine_naive_bayes():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Data
    wine = load_wine()
    X, y = wine.data, wine.target
    target_names = wine.target_names
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.25, random_state=42)
    
    print("--- Wine Category Analysis (Naive Bayes) ---")
    
    # 3. Custom Gaussian Model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    acc_gnb = np.mean(y_pred_gnb == y_test)
    print(f"Gaussian NB Accuracy: {(acc_gnb * 100):.2f}%")
    
    # 4. Custom Multinomial Model
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)
    acc_mnb = np.mean(y_pred_mnb == y_test)
    print(f"Multinomial NB Accuracy: {(acc_mnb * 100):.2f}%")
    
    # 5. Validation with Sklearn
    sk_gnb = SklearnGNB().fit(X_train, y_train)
    sk_acc_gnb = accuracy_score(y_test, sk_gnb.predict(X_test))
    
    sk_mnb = SklearnMNB().fit(X_train, y_train)
    sk_acc_mnb = accuracy_score(y_test, sk_mnb.predict(X_test))
    
    print(f"\n--- Sklearn Validation ---")
    print(f"Sklearn Gaussian NB Accuracy: {(sk_acc_gnb * 100):.2f}%")
    print(f"Sklearn Multinomial NB Accuracy: {(sk_acc_mnb * 100):.2f}%")
    
    if np.isclose(acc_gnb, sk_acc_gnb) and np.isclose(acc_mnb, sk_acc_mnb):
        print("✅ Success! Custom models match Sklearn results exactly.")

    # 6. Data Visualization
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(X, columns=wine.feature_names)
    df['target'] = y
    
    # Plotting Alcohol vs Malic Acid distribution by category
    sns.scatterplot(data=df, x='alcohol', y='malic_acid', hue='target', palette='viridis')
    plt.title("Wine Categories Distribution (Alcohol vs Malic Acid)"); plt.grid(True)
    plt.savefig(os.path.join(current_dir, "example_wine_analysis.png"))
    print(f"\nVisual Analysis Plot generated in: {current_dir}")

    # 7. Sample Predictions
    print(f"\n--- Sample Predictions (Gaussian) ---")
    sample_indices = [0, 15, 30]
    for idx in sample_indices:
        test_sample = X_test[idx]
        true_class = y_test[idx]
        pred_class = gnb.predict([test_sample])[0]
        print(f"Sample {idx} -> Predicted: {target_names[pred_class]}, True: {target_names[true_class]}")

if __name__ == '__main__':
    run_wine_naive_bayes()
