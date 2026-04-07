import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.core.linear_regression import LinearRegression
from utils.data_preprocessing import clean_data

def run_canada_income_exercise():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/canada_per_capita_income.csv")
    df = pd.read_csv(data_path, sep='\t')
    df = clean_data(df)
    
    # Target column name: "per capita income (US$)" 
    target_col = [col for col in df.columns if 'income' in col][0]
    X, y = df[['year']].values, df[target_col].values
    
    # 2. Custom Model Training
    model = LinearRegression()
    model.fit(X, y)
    
    test_years = [2020, 2025, 2030]
    results = {}
    
    print("--- Canada Per Capita Income Analysis (Linear Regression) ---")
    for year in test_years:
        pred = model.predict([[year]]).item()
        results[year] = pred
        print(f"Predicted Income for {year}: ${pred:,.2f}")
    
    # 4. Validation with Sklearn
    sk_model = SklearnLR().fit(X, y)
    print(f"\n--- Sklearn Validation ---")
    sk_pred_2020 = sk_model.predict([[2020]]).item()
    print(f"Sklearn Prediction for 2020: ${sk_pred_2020:,.2f}")
    
    if np.isclose(results[2020], sk_pred_2020):
        print("✅ Success! Custom model matches Sklearn results exactly.")
    
    # 5. Save Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df['year'], df[target_col], color='red', marker='+', label='Historical Data')
    plt.plot(df['year'], model.predict(X), color='blue', label='Linear Regression Line')
    
    # Plotting our predicted points
    plt.scatter(test_years, list(results.values()), color='green', marker='o', label='Predictions')
    
    plt.xlabel("Year")
    plt.ylabel("Income (US$)")
    plt.title("Canada Per Capita Income Forecaster")
    plt.legend()
    plt.grid(True)
    
    plots_dir = os.path.join(current_dir, "../../plots/linear_regression")
    os.makedirs(plots_dir, exist_ok=True)
    plot_file = os.path.join(plots_dir, "canada_income_analysis.png")
    plt.savefig(plot_file)
    print(f"Visualization saved to: {os.path.abspath(plot_file)}")

if __name__ == "__main__":
    run_canada_income_exercise()
