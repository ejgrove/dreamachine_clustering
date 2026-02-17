# Clustering Analysis of Dreamachine Images

Unsupervised clustering analysis of drawings after stroboscopically induced visual hallucinations.

Paper title: "A Large-Scale Computer-Vision Mapping of the Geometric Structures of Stroboscopically-Induced Visual Hallucinations" (2026) Ethan J. Grove, Trevor Hewitt, Anil K. Seth, Fiona Macpherson, David J. Schwartzman.

## Pipeline
1. Dataset and preprocessing
2. DINOv2 embedding
3. Clustering and Analysis

## Dataset and Preprocessing
[data/image_preprocessing_labels.xlsx](data/image_preprocessing_labels.xlsx) includes per-image exclusion labels (duplicates, corrupted files, non-drawings). [notebooks/image_preprocessing.ipynb](notebooks/image_preprocessing.ipynb) filters and preprocesses images into a clean directory.

## DINOv2 Embedding
[notebooks/DINOv2_embedding.ipynb](notebooks/DINOv2_embedding.ipynb) generates embeddings from preprocessed images. The main output is [embeddings/feature_vectors_dict_DM_full_dino_48_augmentations_05092024.pt](embeddings/feature_vectors_dict_DM_full_dino_48_augmentations_05092024.pt).

## Clustering and Analysis
[notebooks/dm_cluster_main.ipynb](notebooks/dm_cluster_main.ipynb) runs PCA, UMAP, and HDBSCAN for clustering and figure generation. Cluster metadata is in [data/cluster_labels.xlsx](data/cluster_labels.xlsx). [notebooks/location_attendance_analysis.ipynb](notebooks/location_attendance_analysis.ipynb) provides statistical analysis of cluster relationships with geographic location and attendance data.

## Setup
1. Create conda environment from the environment file:
   ```bash
   conda env create -f requirements/environment.yml
   conda activate dreamachine_clustering
   ```
   
2. Configure paths in each notebook:
   - Edit `base_dir` to point to this repository
   - Edit `image_dir` to point to preprocessed images directory
   - *The attendance data for the Dreamachine (`attendance_path`) is not publicly available.*
   
3. Ensure embeddings exist at [embeddings/feature_vectors_dict_DM_full_dino_48_augmentations_05092024.pt](embeddings/feature_vectors_dict_DM_full_dino_48_augmentations_05092024.pt).

## Repository Layout
- [notebooks/](notebooks) analysis notebooks
- [utils.py](utils.py) shared helper functions
- [data/](data) cluster labels and drawing metadata
- [embeddings/](embeddings) saved feature vectors
- [outputs/](outputs) cached models and files.
- [clusters/](clusters) images grouped by cluster assignment

## Clustered Drawings
The full set of preprocessed drawings sorted by cluster is availabe here: [/outputs/cluster_images.pdf](/outputs/cluster_images.pdf). 

## Reproducibility
Environment specifications are captured in [requirements/environment.yml](requirements/environment.yml). The DINOv2 embedding notebook has separate dependencies; see [requirements/DINOv2_embedding_requirements.txt](requirements/DINOv2_embedding_requirements.txt).

## TO DO:
- upload preprocessed image dataset and add to readme 