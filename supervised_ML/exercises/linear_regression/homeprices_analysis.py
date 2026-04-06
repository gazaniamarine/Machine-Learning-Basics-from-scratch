import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Adjust path to reach core and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.models.linear_regression import LinearRegression
from utils.data_preprocessing import fix_missing_values, clean_data

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/homeprices.csv")
    
    # 1. Load data
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    # Preprocess missing values with median
    df = fix_missing_values(df, strategy='median')
    
    # 2. Features and Target
    X = df[['area', 'bedrooms', 'age']].values
    y = df['price'].values
    
    # 3. Model Training
    model = LinearRegression()
    model.fit(X, y)
    
    print("--- Home Prices Multivariate Analysis (From Scratch) ---")
    
    # 4. Predictions
    test_cases = [
        [3000, 3, 40],
        [2500, 4, 5]
    ]
    
    for case in test_cases:
        prediction = model.predict([case])
        print(f"Predicted Price (Area={case[0]}, Bed={case[1]}, Age={case[2]}): ${prediction.item():,.2f}")
        
    print("-" * 30)
    print("Model Parameters:")
    print(f"Intercept: {model.bias:.4f}")
    print(f"Weights: {model.weights}")

if __name__ == "__main__":
    main()
