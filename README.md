# Clustering Analysis of Dreamachine Images

This repository contains code for unsupervised clustering analysis of Dreamachine images using DINOv2 features, UMAP, and HDBSCAN.

## Setup

1. **Edit `config.yaml`** with paths to your data:
   - `base_dir`: Path to this clustering folder
   - `images`: Path to the Dreamachine images directory
   - `labels`: Path to cluster labels spreadsheet
   - `metadata`: Path to metadata spreadsheets

2. **Ensure feature vectors exist** at: `data/feature_vectors_dict_DM_full_dino_48_augmentations_05092024.pt`

## Files

- **`cluster_labels.ipynb`**: Main clustering analysis notebook (PCA, UMAP, HDBSCAN)
- **`utils.py`**: Utility functions for visualization and analysis
- **`website.py`**: Export functions for web visualization
- **`data/`**: Feature vectors and metadata
- **`outputs/`**: Generated models and embeddings
- **`clusters/`**: Images organized by cluster assignment

# TODO
- NEED to add environment to config.yaml