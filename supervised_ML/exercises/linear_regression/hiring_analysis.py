import os
import sys
import pandas as pd
import numpy as np

# Adjust path to reach core and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.models.linear_regression import LinearRegression
from utils.data_preprocessing import fix_missing_values, words_to_numbers, clean_data

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/hiring.csv")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    # 2. Preprocessing
    df['experience'] = df['experience'].fillna('zero')
    df = words_to_numbers(df, ['experience'])
    df = fix_missing_values(df, strategy='median')
    
    X = df[['experience', 'test_score', 'interview_score']].values
    y = df['salary'].values
    
    # 3. Model Training
    model = LinearRegression()
    model.fit(X, y)
    
    # 4. Predictions for future candidates
    print("--- Hiring Salary Analysis (From Scratch) ---")
    
    # Candidate 1: 2 yr experience, 9 test score, 6 interview score
    pred1 = model.predict([[2, 9, 6]])
    print(f"Prediction 1 (2yr, 9 test, 6 interview): ${pred1.item():,.2f}")
    
    # Candidate 2: 12 yr experience, 10 test score, 10 interview score
    pred2 = model.predict([[12, 10, 10]])
    print(f"Prediction 2 (12yr, 10 test, 10 interview): ${pred2.item():,.2f}")

if __name__ == "__main__":
    main()
