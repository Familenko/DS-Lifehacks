import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(self, X, y, scaler='StandardScaler'):        
        self.X = X
        self.y = y
        self.scaler = None

        if scaler == 'StandardScaler':
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X)
        elif scaler == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
            self.X = self.scaler.fit_transform(self.X)


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        def to_tensor(x):
            if isinstance(x, pd.DataFrame):
                return torch.tensor(x.iloc[idx].values, dtype=torch.float32)
            elif isinstance(x, pd.Series):
                return torch.tensor(x.iloc[idx], dtype=torch.float32)
            elif isinstance(x, np.ndarray):
                return torch.tensor(x[idx], dtype=torch.float32)
            elif isinstance(x, torch.Tensor):
                return x[idx].float()
            elif isinstance(x, list):
                return torch.tensor(x[idx], dtype=torch.float32)
            else:
                raise TypeError(f"Unsupported type: {type(x)}")
        
        X = to_tensor(self.X)
        y = to_tensor(self.y)

        return X, y
