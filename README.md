# Clustering Analysis of Dreamachine Images

Pipeline for the unsupervised clustering analysis of drawings post induction of stroboscopic visual hallucinations in "A Large-Scale Computer-Vision Mapping of the Geometric Structures of Stroboscopically-Induced Visual Hallucinations" (2026) Ethan J. Grove, Trevor Hewitt, Anil K. Seth, Fiona Macpherson, David J. Schwartzman.

## Pipeline
1. Dataset and preprocessing
2. DINOv2 embedding
3. Clustering and Analysis

## Dataset and Preprocessing
- [data/image_preprocessing_labels.xlsx](data/image_preprocessing_labels.xlsx): Contains per-image exclusion labels (duplicates, corrupted files, non-drawings) and location data
- [notebooks/image_preprocessing.ipynb](notebooks/image_preprocessing.ipynb): Filters and preprocesses drawings.
- The Dreamachine drawing dataset has no yet been made publicly available. 

## DINOv2 Embedding
- [notebooks/DINOv2_embedding.ipynb](notebooks/DINOv2_embedding.ipynb): Generates embeddings from preprocessed drawings.
- [embeddings/feature_vectors_dict_DM_full_dino_48_augmentations_05092024.pt](embeddings/feature_vectors_dict_DM_full_dino_48_augmentations_05092024.pt): Drawing DINOv2 embeddings

## Clustering and Analysis
- [notebooks/dm_cluster_main.ipynb](notebooks/dm_cluster_main.ipynb): Runs PCA, UMAP, and HDBSCAN for clustering and figure generation
- [notebooks/location_attendance_analysis.ipynb](notebooks/location_attendance_analysis.ipynb): Runs  analysis of cluster relationships with geographic location and attendance data
- [data/cluster_labels.xlsx](data/cluster_labels.xlsx): Contains cluster label data

## Setup
1. Create conda environment from the environment file:
   ```bash
   conda env create -f requirements/environment.yml
   conda activate dreamachine_clustering
   ```
   
2. Configure paths in each notebook:
   - Edit `base_dir` to point to this repository
   - Edit `image_dir` to point to preprocessed images directory `dreamachine_drawings_preprocessed/`
   - *The attendance data for the Dreamachine (`attendance_path`) is not publicly available.*

## Expected Outputs
Running the notebooks generates:
- Cached UMAP embeddings and HDBSCAN models in `outputs/cache/`
- Cluster visualizations and figures (Figures 3-5 in paper)
- PDF with all clustered drawings sorted by category

## Repository Layout
- [notebooks/](notebooks) analysis notebooks
- [utils.py](utils.py) shared helper functions
- [data/](data) cluster labels and drawing metadata
- [embeddings/](embeddings) saved feature vectors
- [outputs/](outputs) cached models and files.
- [clusters/](clusters) images grouped by cluster assignment

## Clustered Drawings
Preprocessed drawings sorted by cluster: [/outputs/cluster_images.pdf](/outputs/cluster_images.pdf). 

## Reproducibility
Environment specifications are captured in [requirements/environment.yml](requirements/environment.yml). The DINOv2 embedding notebook has separate dependencies; see [requirements/DINOv2_embedding_requirements.txt](requirements/DINOv2_embedding_requirements.txt).

## TO DO:
