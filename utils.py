import numpy as np

import constants

def load_single_video(vid):
    x_path = constants.DATA_PATH / (vid + constants.EMBED_SUFFIX)
    y_path = constants.DATA_PATH / (vid + constants.LABELS_SUFFIX)
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

def load_multiple_videos(custom_list=None):
    vids = constants.VIDS if not custom_list else custom_list
    embeddings, labels = zip(*[load_single_video(vid) for vid in vids])
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels

def load_all_excluding(vid):
    idx = constants.VIDS.index(vid)
    vids = constants.VIDS[:idx] + constants.VIDS[idx+1:]
    x, y = load_multiple_videos(vids)
    x_test, y_test = load_single_video(vid)
    return x, x_test, y, y_test