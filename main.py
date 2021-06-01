import pytorch_lightning as pl

from data import LipReadingData
from model import LipReader

if __name__ == '__main__':
    dm = LipReadingData(batch_size=32)
    dm.setup('fit') 

    model = LipReader(hidden_size=32, num_layers=1, dropout_rate=0, bidirectional=True, pos_weight=dm.pos_weight)
    trainer = pl.Trainer()
    trainer.fit(model, dm)
