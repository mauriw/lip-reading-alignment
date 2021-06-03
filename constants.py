from pathlib import Path
from enum import Enum

EMBED_LEN = 512
SEQ_LEN = 32
DATA_PATH = Path.cwd() / 'data' / 'embeddings'
EMBED_SUFFIX = '_final_embeddings.npy'
LABELS_SUFFIX = '_embedding_labels.npy'
VIDS = ['5l4cA8zSreQ','pNttjJUtkA4', 'GjYt5uQPu8o', 'WFImhWa5oUw', 'urKhVssiygA', 'CLWRclarri0', 'j7RsRnYlz7I']

class DataSetup(Enum):
    ALL = 1
    SINGLE = 2
    EXCLUDE = 3
