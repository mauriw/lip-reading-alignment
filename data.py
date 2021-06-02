import torch 
import pytorch_lightning as pl
import constants
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from sklearn.model_selection import train_test_split

class LipReadingData(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.x = np.random.randn(100, constants.SEQ_LEN, constants.EMBED_LEN) # 100 datapoints of 32 consecutive 512-d embeddings
        self.y = np.random.randint(2, size=(100, constants.SEQ_LEN)) # 100 datapoints of 32 consecutive labels
        self.batch_size = batch_size
        self.pos_weight = self.get_pos_weight()

    def get_pos_weight(self):
        # Metric used in the loss function to offset our model's low prevalence
        labels = self.y.ravel()
        num_positive = labels.sum()
        num_negative = len(labels) - num_positive
        pos_weight = num_positive / num_negative
        return torch.tensor([pos_weight])

    def setup(self, stage):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
        self.train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        self.val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        self.test = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)