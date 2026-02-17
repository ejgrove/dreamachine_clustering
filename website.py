import json
import numpy as np
import pickle as pkl
import pandas as pd
import utils
from pathlib import Path

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

base_dir = Path(__file__).resolve().parent
path_to_save = base_dir / "outputs" / "cache"

# HDBSCAN model path
hdbscan_path = 'hdbscan_l2norm_90pca_6components_100nn_0dist_cosine_42randseed_100_minclustsize_22minsamples.pkl'

with open(path_to_save / hdbscan_path, 'rb') as file:
    hdbscan_model = pkl.load(file)

hdbscan_model.labels_ = utils.relabel_by_size(hdbscan_model.labels_)

# UMAP embeddings path
umap_2d_150_path = "umap_embeddings_2d_150nn.pkl"

with open(path_to_save / umap_2d_150_path, "rb") as file:
    umap_2d_150 = pkl.load(file)

# Labels
labels_path = base_dir / "data" / "cluster_labels.xlsx"
labels_df = pd.read_excel(labels_path, sheet_name='Sheet1')

labels_df['cluster'] = pd.to_numeric(labels_df['cluster'], errors='coerce').round().astype('Int64')
labels_df = labels_df.dropna(subset=['cluster']).copy()
labels_df['cluster'] = labels_df['cluster'].astype(int)
labels_df['category'] = labels_df['category'].astype(str).str.strip().str.lower()

data = build_embeddings_scattergl_demo(
    umap=umap_2d_150,
    labels=hdbscan_model.labels_,
    labels_df=labels_df,
)

with open("dm_data.json", "w") as file:
    json.dump(data, file)
print("Data saved to dm_data.json")
