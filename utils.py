import numpy as np
from pathlib import Path

# TODO:
#   - separate embeddings and labels into 2 folders
#   - iterate over both, and return concatenated embeddings and labels

def load_data(vid):
    x_path = Path.cwd() / 'data' / 'embeddings' / (vid + '_final_embeddings.npy')
    y_path = Path.cwd() / 'data' / 'embeddings' / (vid + '_embedding_labels.npy')
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y
