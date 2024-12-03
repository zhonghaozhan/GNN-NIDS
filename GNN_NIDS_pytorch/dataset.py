import torch
from torch_geometric.data import Data, Dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

class IDSDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(IDSDataset, self).__init__(root, transform, pre_transform)
        self.data_list = []
        self._process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(i) for i in range(len(self.data_list))]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def _process(self):
        # Find the CSV file in the directory, ignoring macOS hidden files
        csv_files = [f for f in os.listdir(self.root) if f.endswith('.csv') and not f.startswith('._')]
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.root}")
        
        csv_file = os.path.join(self.root, csv_files[0])
        print(f"Loading dataset from {csv_file}")
        
        # Load the CSV file with low_memory=False to avoid DtypeWarning
        df = pd.read_csv(csv_file, low_memory=False)
        
        # Convert all object (string) columns to numeric
        for column in df.columns:
            if df[column].dtype == 'object':
                try:
                    # First try to convert to numeric
                    df[column] = pd.to_numeric(df[column], errors='raise')
                except (ValueError, TypeError):
                    # If that fails, use label encoding
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column].astype(str))
        
        # Ensure all data is float32
        df = df.astype('float32')
        
        # Extract features and labels
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        
        # Check for NaNs in the dataset
        if np.any(np.isnan(features)):
            raise ValueError("NaN values found in features")
        if np.any(np.isnan(labels)):
            raise ValueError("NaN values found in labels")
        
        # Normalize features
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_std[features_std == 0] = 1  # Prevent division by zero
        features = (features - features_mean) / features_std
        
        print(f"Feature shape: {features.shape}")
        print(f"Number of unique labels: {len(np.unique(labels))}")
        print(f"Labels distribution: {np.unique(labels, return_counts=True)}")
        
        # Convert to PyTorch tensors
        x = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        # Create edge indices using a more sophisticated approach
        edge_index = self._create_edge_index(len(df))
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        self.data_list.append(data)

    def _create_edge_index(self, num_nodes):
        # Create a more structured graph based on temporal and feature similarity
        edges = []
        window_size = 100  # Consider temporal relationships within this window
        
        for i in range(num_nodes):
            # Connect to temporal neighbors
            start_idx = max(0, i - window_size // 2)
            end_idx = min(num_nodes, i + window_size // 2)
            
            for j in range(start_idx, end_idx):
                if i != j:
                    edges.append([i, j])
                    edges.append([j, i])  # Add reverse edge for undirected graph
        
        edge_index = torch.LongTensor(edges).t().contiguous()
        return edge_index
