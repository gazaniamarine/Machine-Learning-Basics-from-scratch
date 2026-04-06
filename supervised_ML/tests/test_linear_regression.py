import os
import sys
import pandas as pd
import numpy as np
from sklearn import linear_model

# Adjust system path to reach core and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from supervised_ML.core.linear_regression import LinearRegression
from utils.data_preprocessing import fix_missing_values, words_to_numbers, clean_data

def test_linear_regression():
    """Validates custom Linear Regression implementation against Sklearn."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/hiring.csv")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    # Preprocessing
    df['experience'] = df['experience'].fillna('zero')
    df = words_to_numbers(df, ['experience'])
    df = fix_missing_values(df, strategy='median')
    
    X = df[['experience', 'test_score', 'interview_score']].values
    y = df['salary'].values
    
    # 2. Train Custom Model
    custom_model = LinearRegression()
    custom_model.fit(X, y)
    
    # 3. Train Sklearn Model
    sk_model = linear_model.LinearRegression()
    sk_model.fit(X, y)
    
    # 4. Compare Intercept and Coefficients
    # We use np.allclose to handle small floating point differences
    bias_match = np.allclose(custom_model.bias, sk_model.intercept_)
    weights_match = np.allclose(custom_model.weights, sk_model.coef_)
    
    test_data = [[12, 10, 10]]
    custom_pred = custom_model.predict(test_data)[0]
    sk_pred = sk_model.predict(test_data)[0]
    prediction_match = np.allclose(custom_pred, sk_pred)
    
    print("--- Linear Regression Implementation Validation ---")
    print(f"Bias Check: {'Passed' if bias_match else 'Failed'}")
    print(f"Weights Check: {'Passed' if weights_match else 'Failed'}")
    print(f"Prediction Check: {'Passed' if prediction_match else 'Failed'}")
    print("-" * 30)
    
    if bias_match and weights_match and prediction_match:
        print("Success! Custom implementation matches sklearn perfectly.")
    else:
        print("Test Failed. Results do not match sklearn.")

if __name__ == "__main__":
    test_linear_regression()
