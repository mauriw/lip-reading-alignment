import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import constants

class LipReader(pl.LightningModule):
    def __init__(self, hidden_size, num_layers, dropout_rate, bidirectional, pos_weight):
        super().__init__()
        self.save_hyperparameters() #make sure it works as expected
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size=512, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=num_directions*hidden_size, out_features=1)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # Metrics
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.test_f1 = torchmetrics.F1()

        # For debugging
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        """
        args:
            - x: shape (batch_size, seq_length, 512)
        
        returns:
            - probabilities: shape (batch_size, seq_length, 1)
        """
        batch_size = x.shape[0]
        assert x.shape == (batch_size, constants.SEQ_LEN, constants.EMBED_LEN), x.shape

        output, _ = self.gru(x)
        assert output.shape == (batch_size, constants.SEQ_LEN, self.num_directions * self.hidden_size), output.shape
    
        score = self.linear(output)
        assert score.shape == (batch_size, constants.SEQ_LEN, 1)

        score = torch.squeeze(score)
        assert score.shape == (batch_size, constants.SEQ_LEN)

        return score

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log('train_loss', loss, on_epoch=True)
        self.train_acc(torch.sigmoid(preds), y.int())
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log('val_loss', loss, on_epoch=True)
        self.valid_acc(torch.sigmoid(preds), y.int())
        self.log('val_acc', self.valid_acc, on_step=True, prog_bar=True, on_epoch=True)
        self.log('hp_metric', self.valid_acc)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_acc(torch.sigmoid(preds), y.int())
        self.test_f1(torch.sigmoid(preds), y.int())
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
       
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())