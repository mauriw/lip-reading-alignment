from pathlib import Path

EMBED_LEN = 512
SEQ_LEN = 32
DATA_PATH = Path.cwd() / 'data' / 'embeddings'
EMBED_SUFFIX = '_final_embeddings.npy'
LABELS_SUFFIX = '_embedding_labels.npy'