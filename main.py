import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import LipReadingData
from model import LipReader
import constants 

if __name__ == '__main__':
    data_setup = constants.DataSetup.SINGLE
    dm = LipReadingData(batch_size=64, data_setup=data_setup)
    pos = dm.get_pos_weight() 
    
    model = LipReader(hidden_size=32, num_layers=1, dropout=0, 
                bidirectional=True, pos_weight=pos, data_setup=data_setup)
    trainer = pl.Trainer(callbacks=[EarlyStopping(monitor='val_loss')])
    trainer.fit(model, dm)
    # trainer.test()


# path = '/Users/mauriw/lip-reading-alignment/lightning_logs/version_24/checkpoints/epoch=14-step=74.ckpt'
# model = LipReader.load_from_checkpoint(path)