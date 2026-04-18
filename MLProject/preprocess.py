"""
Automated Preprocessing Script for Credit Card Fraud Detection
Ported to Workflow-CI for pipeline integration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath='creditcard.csv'):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

def handle_missing_values(df):
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df = df.dropna()
    return df

def handle_duplicates(df):
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
    return df

def scale_features(df):
    scaler = StandardScaler()
    df['Scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['Scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def automate_preprocessing(input_path='creditcard.csv',
                           output_path='creditcard_preprocessed.csv'):
    df = load_data(input_path)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = scale_features(df)
    
    # We save the full preprocessed data for MLflow Project
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data: {output_path} ({len(df)} rows)")
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="creditcard.csv")
    parser.add_argument("--output", type=str, default="creditcard_preprocessed.csv")
    args = parser.parse_args()
    automate_preprocessing(args.input, args.output)
