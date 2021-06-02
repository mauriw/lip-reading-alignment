import numpy as np

import constants

def load_data(vid):
    x_path = constants.DATA_PATH / (vid + constants.EMBED_SUFFIX)
    y_path = constants.DATA_PATH / (vid + constants.LABELS_SUFFIX)
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

def load_all_data():
    vids = ['5l4cA8zSreQ','pNttjJUtkA4', 'GjYt5uQPu8o', 'WFImhWa5oUw', 'urKhVssiygA', 'CLWRclarri0', 'j7RsRnYlz7I']
    embeddings, labels = zip(*[load_data(vid) for vid in vids])
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels