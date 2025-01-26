import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from generator import features  # Import feature list from generator
import time

def convert_to_ids2017_format(df):
    """Convert ML dataset to IDS2017 format."""
    # Create an empty DataFrame with required columns
    required_features = [
        'Source Port', 'Destination Port', 'Bwd Packet Length Min', 'Subflow Fwd Packets',
        'Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Fwd IAT Min', 'Flow IAT Min', 'Flow IAT Mean', 'Bwd Packet Length Std',
        'Subflow Fwd Bytes', 'Flow Duration', 'Flow IAT Std', 'Active Min', 'Active Mean',
        'Bwd IAT Mean', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'ACK Flag Count',
        'Fwd PSH Flags', 'SYN Flag Count', 'PSH Flag Count', 'URG Flag Count'
    ]
    
    # Initialize output DataFrame with required columns
    ids2017_df = pd.DataFrame(columns=required_features + ['Flow ID', 'Source IP', 'Destination IP', 'Attack_type', 'Attack_label'])
    
    # Copy existing columns
    for col in df.columns:
        if col in ids2017_df.columns:
            ids2017_df[col] = df[col]
    
    # Copy IP addresses and Flow ID
    ids2017_df['Flow ID'] = df['Flow ID']
    ids2017_df['Source IP'] = df['Source IP']
    ids2017_df['Destination IP'] = df['Destination IP']
    
    # Fill missing values with 0
    for feature in required_features:
        if feature not in df.columns:
            ids2017_df[feature] = 0
        else:
            ids2017_df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
    
    # Add attack information
    ids2017_df['Attack_type'] = df['Attack_type']
    ids2017_df['Attack_label'] = df['Attack_label']
    
    return ids2017_df

def load_and_split_data(csv_path, train_ratio=0.7, random_state=42):
    """
    Load ML dataset and split into train and eval sets.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Convert to IDS2017 format
    print("Converting to IDS2017 format...")
    df = convert_to_ids2017_format(df)
    
    # Split into train and eval sets
    train_data, eval_data = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df['Attack_type']
    )
    
    print(f"Train set size: {len(train_data)}")
    print(f"Eval set size: {len(eval_data)}")
    
    return train_data, eval_data

def save_dataset(data, output_dir):
    """
    Save dataset to specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data.csv')
    data.to_csv(output_path, index=False)
    print(f"Saved data to {output_path}")
    
    # Print dataset statistics
    print("\nDataset statistics:")
    print("Attack type distribution:")
    print(data['Attack_type'].value_counts())
    print("\nAttack label distribution:")
    print(data['Attack_label'].value_counts())

def main():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'GNN_NIDS_tensorflow/preprocessed_ML_dataset/TRAIN/data.csv')
    output_base = os.path.join(base_dir, 'GNN_NIDS_tensorflow/preprocessed_ML_dataset')
    train_dir = os.path.join(output_base, 'TRAIN')
    eval_dir = os.path.join(output_base, 'EVAL')
    
    # Load and split data
    train_data, eval_data = load_and_split_data(input_path)
    
    # Save datasets
    print("\nSaving training data...")
    save_dataset(train_data, train_dir)
    
    print("\nSaving evaluation data...")
    save_dataset(eval_data, eval_dir)
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()
