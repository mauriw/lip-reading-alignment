import torch 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import utils 
from constants import DataSetup

class LipReadingData(pl.LightningDataModule):
    def __init__(self, batch_size, data_setup: DataSetup):
        super().__init__()
        self.data_setup = data_setup
        self.batch_size = batch_size
        self.setup('fit')

    def get_pos_weight(self):
        # Metric used in the loss function to offset our model's low prevalence
        labels = self.train.tensors[1].ravel()
        num_positive = labels.sum()
        num_negative = len(labels) - num_positive
        pos_weight = num_negative / num_positive  
        return pos_weight.item()

    def setup(self, stage):
        if self.data_setup == DataSetup.ALL or self.data_setup == DataSetup.SINGLE:
            if self.data_setup == DataSetup.SINGLE:
                x, y = utils.load_single_video('pNttjJUtkA4')
            else:
                x, y = utils.load_multiple_videos()
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.50, random_state=42)
        else: # self.data_setup == DataSetup.EXCLUDE:
            x, X_test, y, y_test = utils.load_all_excluding('j7RsRnYlz7I')
            X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=42)
        self.train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        self.val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
        self.test = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        print(self.train.tensors)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)