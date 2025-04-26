import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import sys
import argparse

# Suppress all print statements
import os
import sys

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

with SuppressPrints():
    # 1. Define the model class (same as in training)
    class PricePredictor(nn.Module):
        def __init__(self, input_dim):
            super(PricePredictor, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # 2. Load the training data to get column names and fit scaler
    try:
        df_train = pd.read_csv('ProductPriceIndex_AUD_per_kg.csv')
    except Exception as e:
        # Create a dummy DataFrame with minimal columns
        df_train = pd.DataFrame({
            'productname': ['apple', 'banana', 'orange'],
            'is_in_season': [1, 0, 1],
            'natural_disruption': [0, 0, 1],
            'avg_retail_price': [2.50, 3.20, 1.80]
        })

    # 3. Simple function to predict from a single comma-separated string
    def predict_from_string(data_string):
        """
        Predict price from a comma-separated string of feature values.
        
        Args:
            data_string (str): Comma-separated values matching the training data format
            
        Returns:
            float: Predicted price in AUD per kg
        """
        # Convert string to values
        values = data_string.strip().split(',')
        
        # Create a DataFrame with a single row using the training data's columns
        # Exclude price columns
        feature_cols = [col for col in df_train.columns 
                       if not ('price' in col.lower() or 'retail' in col.lower())]
        
        # If there's a mismatch in number of columns, handle it silently
        if len(values) > len(feature_cols):
            values = values[:len(feature_cols)]
        elif len(values) < len(feature_cols):
            values.extend(['0'] * (len(feature_cols) - len(values)))
        
        # Create a single row DataFrame with as many columns as we can fill
        row_data = {}
        for i, col in enumerate(feature_cols):
            if i < len(values):
                # Convert to appropriate type
                val = values[i]
                if val.lower() in ('true', 'false', ''):
                    # Convert to boolean then to int
                    if val.lower() == 'true':
                        row_data[col] = 1
                    elif val.lower() == 'false':
                        row_data[col] = 0
                    else:  # Empty value
                        row_data[col] = 0
                else:
                    try:
                        row_data[col] = float(val)
                    except ValueError:
                        row_data[col] = val  # Keep as string if can't convert
        
        # Create dataframe
        sample = pd.DataFrame([row_data])
        
        # Fill missing columns with zeros
        for col in feature_cols:
            if col not in sample.columns:
                sample[col] = 0
        
        # 4. Prepare feature data like in the training script
        # Convert categorical columns (like productname) to numeric
        for col in sample.select_dtypes(exclude=['number']).columns:
            sample[col] = 0  # Simple solution: set to 0 for prediction
        
        # Ensure all columns are numeric
        for col in sample.columns:
            if not pd.api.types.is_numeric_dtype(sample[col]):
                sample[col] = 0
        
        # Scale features using same scaler as training
        scaler = StandardScaler()
        X_train = df_train[feature_cols].replace({'True': 1, 'False': 0, True: 1, False: 0}).astype(float).values
        scaler.fit(X_train)
        
        # Transform the sample
        sample_array = sample[feature_cols].values
        scaled_sample = scaler.transform(sample_array)
        
        # 5. Load model and predict
        input_dim = len(feature_cols)
        model = PricePredictor(input_dim)
        model.load_state_dict(torch.load('produce_price_model.pth'))
        model.eval()
        
        with torch.no_grad():
            X = torch.tensor(scaled_sample, dtype=torch.float32)
            prediction = model(X).item()
        
        return round(prediction, 2)

# Example usage: when script is run directly

# Predict price from a comma-separated string (API-like function)
def predict_price(data_string: str) -> float:
    """
    Predict price from a comma-separated string of feature values.
    Args:
        data_string (str): Comma-separated string
    Returns:
        float: Predicted price
    """
    try:
        with SuppressPrints():
            return predict_from_string(data_string)
    except Exception:
        return -1.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict produce prices from CSV-like data', add_help=False)
    parser.add_argument('--data', type=str, 
                        default="True,False,2025,4,25,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False",
                        help='Comma-separated data string for prediction')
    args, _ = parser.parse_known_args()

    price = predict_price(args.data)
    print(price)
