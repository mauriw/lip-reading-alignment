import torch 
import torch.nn as nn
import pytorch_lightning as pl

import constants

class LipReader(pl.LightningModule):
    def __init__(self, hidden_size, num_layers, dropout_rate, bidirectional, pos_weight):
        super().__init__()
        num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size=512, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(in_features=num_directions*hidden_size, out_features=1)
        self.pos_weight = pos_weight

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
        y_hat = self(x)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fn(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())