import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Import custom core model and preprocessing utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from supervised_ML.models.linear_regression import LinearRegression
from utils.data_preprocessing import clean_data

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../../data/canada_per_capita_income.csv")
    
    # 1. Load Data
    df = pd.read_csv(data_path, sep='\t')
    df = clean_data(df)
    
    X = df[['year']].values
    y = df['per capita income (US$)'].values
    
    # 2. Train Custom Model
    model = LinearRegression()
    model.fit(X, y)
    
    # 3. Predict for 2020
    predict_year = 2020
    prediction = model.predict([[predict_year]])
    
    print(f"--- Canada Per Capita Income Analysis ---")
    print(f"Predicted Income for {predict_year}: ${prediction.item():,.2f}")
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+', label='Actual Data')
    plt.plot(df['year'], model.predict(X), color='blue', label='Linear Regression Line')
    plt.xlabel("Year")
    plt.ylabel("Income (US$)")
    plt.title("Canada Income Prediction")
    plt.legend()
    plt.grid(True)
    
    # Save Plot
    plots_dir = os.path.join(current_dir, "../../plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "canada_income_exercise.png"))
    print(f"Plot saved to: {os.path.abspath(os.path.join(plots_dir, 'canada_income_exercise.png'))}")

if __name__ == "__main__":
    main()
