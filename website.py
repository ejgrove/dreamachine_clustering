import json
import math
import numpy as np

def normalize_embeddings(embeddings, min_val=-10, max_val=10):

    min_embedding = np.min(embeddings, axis=0)
    max_embedding = np.max(embeddings, axis=0)
    normalized = (embeddings - min_embedding) / (max_embedding - min_embedding) * (max_val - min_val) + min_val
    return normalized

def build_embeddings_scattergl_demo(umap, labels, labels_df):
    """
    umap      : array-like float32/64 (N,2)
    labels    : array-like ints (N,)        (HDBSCAN labels; -1 = noise)
    labels_df : pandas DF with columns ['cluster','category','description']
    """
    data = {}
    data["label"] = []
    data["projection"] = []
    data["labelNames"] = []
    data["categories"] = []
    data["quantity"] = []
    
    umap = normalize_embeddings(umap, min_val=-10, max_val=10)
    
    for idx, label in enumerate(labels):
        if label < 0:
            label = 0
        data["label"].append(int(label))
        x = float(umap[idx][0])
        y = float(umap[idx][1])
        data["projection"].append([x, y, 0.0])
        
    for _, row in labels_df.iterrows():
        cluster = int(row.cluster)
        data["labelNames"].append(f"{row.description}")
        data["categories"].append(f"{row.category}")
        data["quantity"].append(f"{row.quantity}")

    return data
