import os
import numpy as np
import matplotlib.pyplot as plt
import random
import glasbey
import pandas as pd
from PIL import Image
from adjustText import adjust_text 
from contextlib import ExitStack
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from matplotlib import cm
from sklearn.base import clone
from matplotlib.offsetbox import (AnnotationBbox, OffsetImage)
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.notebook import tqdm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker
from time import time
from collections import defaultdict, Counter

############################################################################################################
# HELPER FUNCTIONS
#############################################################################################################

def relabel_by_size(labels):
    labels = np.array(labels)
    counts = Counter(labels[labels != -1])  # only non -1
    # Sort clusters by size (descending)
    sorted_clusters = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)

    # Build mapping: cluster -> new label
    new_labels_map = {old: i+1 for i, old in enumerate(sorted_clusters)}

    # Apply mapping, keeping -1 untouched
    relabeled = np.array([new_labels_map.get(lbl, -1) for lbl in labels])

    return relabeled

############################################################################################################
# SPRITE
############################################################################################################

def images_to_sprite(data, invert_colors=False):
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    if invert_colors:
        data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data, n

def create_sprite(filenames, image_folder, output_file, image_size=(50, 50), invert_colors=False):
    # Load and resize images
    images = [np.array(Image.open(os.path.join(image_folder, f)).resize(image_size)) for f in filenames]
    images = np.stack(images, axis=0)
    
    # Create sprite image
    sprite_image, n = images_to_sprite(images, invert_colors=invert_colors)
    
    # Save the sprite image
    sprite_image_pil = Image.fromarray(sprite_image)
    sprite_image_pil.save(output_file)

############################################################################################################
# PLOTTING PROJECTIONS
############################################################################################################

def print_pca_variations(vectors, variance_threshold=0.8, show=True, num_components=20, xlim=None, ratio=True):
        # Assuming 'vectors' is your list of feature vectors
        X = np.array(vectors)  # Convert list of vectors to a numpy array

        # Initialize PCA
        pca = PCA(n_components=min(X.shape[0], X.shape[1]), random_state=42)  # Set n_components to the minimum of the number of samples or features
        pca.fit(X)

        # Explained variance
        explained_variance = pca.explained_variance_
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)  # Compute the cumulative sum of explained variance ratios

        if variance_threshold:
                # Find the number of components needed to reach the 0.8 threshold
                num_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1

        if show:
                # Plotting
                plt.figure(figsize=(10, 6))
                
                if ratio:
                        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, label='Individual explained variance')
                else:
                        plt.plot(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, label='Individual explained variance')

                if variance_threshold:
                        # Highlight components under the threshold
                        plt.axvline(x=num_components, color='green', label=f'{variance_threshold} Variance Threshold')
                else:
                        # Highlight components under the threshold
                        plt.axvline(x=num_components, color='green', label=f'{num_components} components')

                if xlim:
                        plt.xlim(0, xlim)
                plt.ylim(0, 0.5*max(explained_variance))
                if ratio:
                        plt.ylabel('Explained variance ratio')
                else: 
                        plt.ylabel('Eigenvalues')
                plt.xlabel('Principal components')
                plt.title('Explained Variance by Principal Components (Scree Plot)')
                plt.legend()
                plt.show()

        vectors_reduced = PCA(n_components=num_components, whiten=False, random_state=42).fit_transform(X)
        vectors_whitened = PCA(n_components=num_components, whiten=True, random_state=42).fit_transform(X)

        # Information about the transformation
        print(f"Original feature vector shape: {X.shape}")
        print(f"Reduced feature vector shape: {vectors_reduced.shape}")

        if variance_threshold:  
                print(f"Number of components to retain (to reach {variance_threshold} variance explained): {num_components}")
        else:
                print(f"Variance explained by {num_components} components: {cumulative_variance[num_components - 1]}")

        return vectors_reduced, vectors_whitened, explained_variance_ratio, num_components


def get_sprite_image(sprite_image, index, image_size, grid_size):
        row = index // grid_size
        col = index % grid_size
        left = col * image_size[0]
        upper = row * image_size[1]
        right = left + image_size[0]
        lower = upper + image_size[1]

        return sprite_image.crop((left, upper, right, lower))
    
def plot_with_images(embedding, 
                     model_dict, 
                     subsample_size=15000, 
                     cleaned=False, 
                     with_filenames=False,
                     figsize=(40, 40), 
                     zoom=0.4):
    
    IMAGE_SIZE = (50, 50)  # Resize images for plotting
    IMAGE_FOLDER = model_dict['image_dir']
    SPRITE_FILE = model_dict['sprite']
    filenames = model_dict['filenames']
    if cleaned:
        filenames = model_dict['filenames_inliers']
    NUM_IMAGES = len(filenames)
    
    if subsample_size < NUM_IMAGES:
        # Subsample the data
        indices = np.random.choice(NUM_IMAGES, subsample_size, replace=False)
        subsample_embedding = embedding[indices]
    else:
        indices = np.arange(NUM_IMAGES)
        subsample_embedding = embedding[indices]

    # Load the sprite image
    sprite_image = Image.open(SPRITE_FILE)
    sprite_width, sprite_height = sprite_image.size
    grid_size = int(np.ceil(np.sqrt(NUM_IMAGES)))

    def get_sprite_image(sprite_image, index, image_size, grid_size):
        row = index // grid_size
        col = index % grid_size
        left = col * image_size[0]
        upper = row * image_size[1]
        right = left + image_size[0]
        lower = upper + image_size[1]
        # Ensure the sprite extraction doesn't go out of bounds
        if right > sprite_width or lower > sprite_height:
            return None
        return sprite_image.crop((left, upper, right, lower))

    fig, ax = plt.subplots(figsize=figsize)
    
    print(subsample_embedding.shape, len(indices))

    for i, (x, y) in enumerate(subsample_embedding):
        img_index = indices[i]
        img_index = filenames.index(filenames[img_index]) if cleaned else img_index
        img = get_sprite_image(sprite_image, img_index, IMAGE_SIZE, grid_size)
        if img is not None:
            img = np.array(img)
            im = OffsetImage(img, zoom=zoom)  # Adjust zoom if needed
            ab = AnnotationBbox(im, (x, y), frameon=False)
            ax.add_artist(ab)
            if with_filenames:
                ax.text(x, y - 5 * zoom, filenames[img_index], fontsize=8, ha='center', va='top', color='black')

    # Set plot limits and labels
    ax.set_xlim(subsample_embedding[:, 0].min()-1, subsample_embedding[:, 0].max()+1)
    ax.set_ylim(subsample_embedding[:, 1].min()-1, subsample_embedding[:, 1].max()+1)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.axis('off')
    plt.show()

############################################################################################################
# CLUSTERING METRICS
############################################################################################################

def silhouette_plot(cluster_labels, vectors, metric='euclidean'):

    plt.figure(figsize=(9, 9))

    # num of clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # SORT CLUSTERS BY NUMBER OF DRAWINGS
    
    # Sort the dictionaries by the length of the lists in their keys
    counts = Counter(cluster_labels)
    # Example: if counts = {2: 4, 0: 3, 1: 3}, 
    # this means cluster "2" has 4 elements, cluster "0" has 3 elements, cluster "1" has 3 elements

    # 2. Sort clusters by size (descending order)
    #    largest -> smallest
    sorted_clusters_by_size = sorted(counts.keys(), key=lambda lbl: counts[lbl], reverse=True)
    # Example: sorted_clusters_by_size might be [2, 0, 1] if cluster 2 has the most elements

    # 3. Build a mapping from old labels to new labels
    #    The most populated cluster gets the new label 0,
    #    the second most populated gets label 1, and so on.
    new_label_map = {}
    for new_lbl, old_lbl in enumerate(sorted_clusters_by_size):
        new_label_map[old_lbl] = new_lbl

    # 4. Generate the new cluster labels
    cluster_labels = np.array([new_label_map[old_lbl] for old_lbl in cluster_labels])

    # The silhouette coefficient can range from -1, 1, but here we use -0.1 to 1
    plt.xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(vectors) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters

    silhouette_avg = silhouette_score(vectors, cluster_labels, metric=metric)
    
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    if -1 in cluster_labels:

        filtered_vectors = [vectors[i] for i in range(len(cluster_labels)) if cluster_labels[i] != -1]
        filtered_labels = [label for label in cluster_labels if label != -1]

        silhouette_avg = silhouette_score(filtered_vectors, filtered_labels, metric=metric)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(vectors, cluster_labels, metric=metric)

    y_lower = 10
    for i in unique_clusters:
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title(f"N={n_clusters} Average Silhouette={silhouette_avg}")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

def print_metrics(cluster_labels, vectors, metric='euclidean'):
    
    sil_score = metrics.silhouette_score(vectors, cluster_labels, metric=metric)
    cal_har_score = metrics.calinski_harabasz_score(vectors, cluster_labels)
    dav_boul_score = davies_bouldin_score(vectors, cluster_labels)

    print(f"Silhouette Score: {round(sil_score, 5)}")
    print(f"Calinski Harabaz Score: {round(cal_har_score, 5)}")
    print(f"Davies Bouldin Score: {round(dav_boul_score, 5)}")
    
def get_neg_silhouette(cluster_labels, vectors):
    
    # label images with negative silhouette scores

    # num of clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    silhouette_avg = silhouette_score(vectors, cluster_labels)
    
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(vectors, cluster_labels)
    
    cluster_neg_silhoutte = []

    for i in sample_silhouette_values:
        if i < 0:
            cluster_neg_silhoutte.append(1)
        else:
            cluster_neg_silhoutte.append(0)

    return cluster_neg_silhoutte


def metrics_over_range(clustering_algo, 
                       vectors, 
                       param_name, 
                       param_range, 
                       model_dict, 
                       metric='euclidean', 
                       sse=False, 
                       show_plot=True,
                       show_best=True,
                       show_silhouette=False, 
                       show_clusters=False, 
                       save=False, 
                       to_return=False):
    
    # Ensure the save directory exists
    if not os.path.exists(save):
        os.makedirs(save)
    
    scores = defaultdict(list) # Dictionary to store scores
    best_param = defaultdict(list) # Dictionary to store best parameters
    
    vectors = np.array(vectors)
    
    if not show_silhouette:
        param_loop = tqdm(param_range, leave=False)
    else:
        param_loop = param_range

    for param_value in param_loop:
        # Clone the original algorithm with the new parameter value
        algo = clone(clustering_algo)
        setattr(algo, param_name, param_value)  # Set the parameter
        cluster_fit = algo.fit(vectors)
        cluster_labels = cluster_fit.labels_

        # Compute metrics
        scores['average silhouette score'].append(metrics.silhouette_score(vectors, cluster_labels, metric=metric))
        scores['cal_har_score'].append(metrics.calinski_harabasz_score(vectors, cluster_labels))
        scores['dav_boul_score'].append(metrics.davies_bouldin_score(vectors, cluster_labels))
        
        if show_clusters:
            print_clusters_pdf_centroid(cluster_labels,
                                        vectors,
                                        model_dict,
                                        random_sample=80,
                                        random_centroids=20,
                                        save=os.path.join(save,f'{param_value}_clusters.pdf'))
            
        if sse:
            scores['sse'].append(cluster_fit.inertia_)

        if show_silhouette:
            silhouette_plot(cluster_labels, vectors)

    # Plotting
    for key, value in scores.items():

        plt.figure(figsize=(10, 5))
        
        if key in ['average silhouette score', 'cal_har_score']:
            if show_best:
                best_param[key] = param_range[np.argmax(value)]
                print(f"Best {key}: {param_range[np.argmax(value)]}")
                plt.axvline(x=param_range[np.argmax(value)], color='red', linestyle='--')
        
        elif key == 'dav_boul_score':
            if show_best:
                best_param[key] = param_range[np.argmin(value)]
                print(f"Best {key}: {param_range[np.argmin(value)]}")
                plt.axvline(x=param_range[np.argmin(value)], color='red', linestyle='--')
        plt.plot(param_range, value)
        plt.xlabel('k clusters')
        plt.ylabel(key)
        plt.xticks([p for p in param_range if p % 2 == 0])
        plt.grid(True)
        if save:
            plt.savefig(os.path.join(save, f'{key}_vs_{param_name}.png'))

        if show_plot:
            plt.show()
        else:
            plt.close()

    return_variables = {"best_param": best_param, 
                        "scores": scores}

    if to_return:
        return return_variables[to_return]

def _plot_metric(x, mean, std, title, ylabel, lower_is_better=False, logy=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, mean, marker='o', label='Mean')
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, label='±1 SD')
    ax.set_xlabel('k clusters')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)
    if lower_is_better:
        ax.invert_yaxis()
    if logy:
        ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    return fig

class ShowInstead:
    def savefig(self, fig): 
        plt.show(block=False)   # non-blocking show; drop block=True if you want blocking
    def __enter__(self): return self
    def __exit__(self, *exc): pass

def plot_metric_averaged_seeds(embeddings, 
                               parameters,
                               model_dict,
                               k_range,
                               average_embeddings=False,
                               n_seeds=10,
                               pdf_save=False,
                               show_plot=True,
                               ):
    
    def average_and_plot(silhouette_scores,
                         cal_har_scores,
                         dav_boul_scores,
                         sse_scores,
                         k_range=k_range,
                         parameter=None):

        silhouette_scores = np.array(silhouette_scores)
        silhouette_mean, silhouette_std = silhouette_scores.mean(0), silhouette_scores.std(0)

        cal_har_scores = np.array(cal_har_scores)
        ch_mean, ch_std = cal_har_scores.mean(0), cal_har_scores.std(0)

        dav_boul_scores = np.array(dav_boul_scores)
        db_mean, db_std = dav_boul_scores.mean(0), dav_boul_scores.std(0)

        sse_scores = np.array(sse_scores)
        sse_mean, sse_std = sse_scores.mean(0), sse_scores.std(0)

        # --- Silhouette ---
        fig = _plot_metric(k_range, silhouette_mean, silhouette_std,
                        title=f'Silhouette vs k (parameter {parameter})', ylabel='Silhouette')
        pdf_all.savefig(fig); pdf_sil.savefig(fig); plt.close(fig)

        # --- Calinski–Harabasz ---
        fig = _plot_metric(k_range, ch_mean, ch_std,
                        title=f'Calinski–Harabasz vs k (parameter {parameter})', ylabel='Calinski–Harabasz')
        pdf_all.savefig(fig); pdf_ch.savefig(fig); plt.close(fig)

        # --- Davies–Bouldin ---
        fig = _plot_metric(k_range, db_mean, db_std,
                        title=f'Davies–Bouldin vs k (parameter {parameter})', ylabel='Davies–Bouldin')
        pdf_all.savefig(fig); pdf_db.savefig(fig); plt.close(fig)

        # --- SSE ---
        fig = _plot_metric(k_range, sse_mean, sse_std,
                        title=f'SSE vs k (parameter {parameter})', ylabel='SSE', logy=False)
        pdf_all.savefig(fig); pdf_sse.savefig(fig); plt.close(fig)
    
    # file paths
    pdf_all_path = "cluster_metrics_all_variances.pdf"
    pdf_sil_path = "metric_silhouette_all_variances.pdf"
    pdf_ch_path  = "metric_calinski_harabasz_all_variances.pdf"
    pdf_db_path  = "metric_davies_bouldin_all_variances.pdf"
    pdf_sse_path = "metric_sse_all_variances.pdf"

    with ExitStack() as stack:
        if pdf_save:
            opener = lambda p: stack.enter_context(PdfPages(p))
        else:
            opener = lambda p: stack.enter_context(ShowInstead())

        pdf_all = opener(pdf_all_path)
        pdf_sil = opener(pdf_sil_path)
        pdf_ch  = opener(pdf_ch_path)
        pdf_db  = opener(pdf_db_path)
        pdf_sse = opener(pdf_sse_path)

        if average_embeddings:
            silhouette_scores, cal_har_scores, dav_boul_scores, sse_scores = [], [], [], []

        for parameter in parameters:
            print(f"Processing parameter: {parameter}")
            vectors = embeddings[parameter]

            if not average_embeddings:
                silhouette_scores, cal_har_scores, dav_boul_scores, sse_scores = [], [], [], []

            # run seeds in both modes
            for random_s in tqdm(range(n_seeds)):
                kmeans = KMeans(init="k-means++", n_init=10, max_iter=1000, random_state=random_s)

                scores = metrics_over_range(
                    kmeans, vectors, 'n_clusters', k_range, model_dict,
                    show_silhouette=False, show_plot=False, sse=True,
                    show_clusters=False, save=False, show_best=False, to_return='scores'
                )

                silhouette_scores.append(scores['average silhouette score'])
                cal_har_scores.append(scores['cal_har_score'])
                dav_boul_scores.append(scores['dav_boul_score'])
                sse_scores.append(scores['sse'])

            if not average_embeddings: 
                average_and_plot(silhouette_scores,
                                 cal_har_scores,
                                 dav_boul_scores,
                                 sse_scores,
                                 parameter=parameter)
                
        if average_embeddings:
            average_and_plot(silhouette_scores,
                             cal_har_scores,
                             dav_boul_scores,
                             sse_scores,
                             parameter="averaged")
            
    if pdf_save:
        print("Saved:", pdf_all_path, pdf_sil_path, pdf_ch_path, pdf_db_path, pdf_sse_path)
    else:
        print("pdf_save=False — figures were shown interactively.")



def fit_and_evaluate(km, X, name=None, n_runs=10):
    name = km.__class__.__name__ if name is None else name
    
    print(name)
    
    X=np.array(X)

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
       
        scores['SD'].append(
            SD(X, km.labels_, k=22, centers_id=None, alg_noise='bind',
               centr='mean', nearest_centr=True, metric='euclidean'))
        cluster_ids, cluster_sizes = np.sort(np.unique(km.labels_, return_counts=True))
        print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)
        
############################################################################################################
# CLUSTERING
############################################################################################################

def clusters_dict(labels, vectors, filenames):
    labels = list(labels)
    # Group filenames by cluster
    clustered_images = {}
    for filename, label in zip(filenames, labels):
        if label not in clustered_images:
            clustered_images[label] = []
        clustered_images[label].append(filename)
        
    return clustered_images

def print_clusters(clusters_,
                   vectors,
                   model_dict,
                   show_img=True,
                   cleaned=False,
                   random_sample=None,
                   sort=True,
                   num_images=10,
                   print_filename=False,
                   print_clusters_df=False,
                   negative_silhouette=None,
                   silhouette = False,
                   save=False):
    
    if cleaned:
        filenames = model_dict['filenames_inliers']
    else:
        filenames = model_dict['filenames']
        
    image_dir = model_dict['image_dir']

    clusters = clusters_dict(clusters_, vectors, filenames)
    
    sample_silhouette_values = silhouette_samples(vectors, clusters_, metric='euclidean')

    if sort:
        clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        
    if print_clusters_df:
        clusters_df = pd.DataFrame(columns=['Cluster', 'N', 'Silhouette Score'])

    for cluster, images in clusters:
        
        avg_cluster_silhouette = np.mean(sample_silhouette_values[clusters_ == cluster])
        
        print(f'CLUSTER: {cluster}')
        print(f'N: {len(images)}')
        print(f'Silhouette Score: {avg_cluster_silhouette}')
        print('-'*50)
        
        if print_clusters_df:
            new_row = pd.DataFrame([[cluster, len(images), avg_cluster_silhouette]], columns=['Cluster', 'N', 'Silhouette Score'])
            clusters_df = pd.concat([clusters_df, new_row])
            
        if random_sample:
            if len(images) > random_sample:
                images = random.sample(images, k=random_sample)

        if (cluster != -1) and show_img:
            
                num_rows = (random_sample - 1) // num_images + 1

                plt.figure(figsize=(num_images, num_rows))

                for idx, filename in enumerate(images):
                    
                    img_idx = filenames.index(filename)

                    plt.subplot(num_rows , num_images, idx+1)
                    plt.axis('off')
                    img = Image.open(os.path.join(image_dir, str(filename)))
                    print(f'{idx+1}:  {filename}')
                    if negative_silhouette:
                        neg_sil = negative_silhouette[img_idx]
                        if neg_sil:
                            # add red border
                            current_axis = plt.gca()
                            current_axis.add_patch(patches.Rectangle((0, 0), img.width, img.height, edgecolor='red', facecolor='none', linewidth=2))
                            
                    plt.imshow(img)
                    if print_filename==True:
                        plt.title(str(idx+1), fontsize = 6)

                plt.tight_layout()
                
                if save:
                    plt.savefig(os.path.join(save, f'cluster_{cluster}.png'))
                    
                plt.show()
    
    if print_clusters_df:
        print(clusters_df)
        return clusters_df
    
def print_centroid_clusters(kmeans, 
                            vectors,
                            model_dict,
                            show_img=True,
                            cleaned=False,
                            sort=True,
                            show=True,
                            num_images=10,
                            print_filename=False,
                            save_filename=False,
                            pdf_save=False):
    
    # Get cluster labels and compute distances to centroids
    clusters_ = kmeans.labels_
    distances_to_centroid = kmeans.transform(vectors)[np.arange(len(vectors)), clusters_]

    # Load filenames
    if cleaned:
        filenames = model_dict['filenames_inliers']
    else:
        filenames = model_dict['filenames']
    image_dir = model_dict['image_dir']

    # Group filenames and distances by cluster
    clustered_data = defaultdict(list)
    for filename, distance, label in zip(filenames, distances_to_centroid, clusters_):
        clustered_data[label].append((distance, filename))

    # Sort clusters by the number of elements and sort items within each cluster by distance
    sorted_clustered_data = {
        label: sorted(items, key=lambda x: x[0])  # Sort items in each cluster by distance
        for label, items in sorted(clustered_data.items(), key=lambda x: len(x[1]))  # Sort clusters by size
    }

    # Extract sorted filenames and distances
    clustered_filenames = {label: [filename for _, filename in items] for label, items in sorted_clustered_data.items()}
    clustered_distances = {label: [distance for distance, _ in items] for label, items in sorted_clustered_data.items()}

    with PdfPages(save_filename) as pdf:
        for clust_idx, (cluster, images) in enumerate(reversed(list(clustered_filenames.items()))):
    
            print(f'CLUSTER: {clust_idx}')
            print(f'N: {len(images)}')
            print('-'*50)
            
            total_images = len(images)
            total_images_shown = 0

            num_columns = 10
            num_rows = (num_images - 1) // num_columns + 1
            
            page_idx = 1
            plt.figure(figsize=(num_columns, num_rows+1))
            plt.axis('off')
            plt.suptitle(f'CLUSTER {clust_idx + 1}', y=0.91, fontsize=12)

            for idx, filename in enumerate(images):
                        
                    distance = np.round(clustered_distances[cluster][idx], 5)

                    plt.subplot(num_rows, num_columns, idx+1-total_images_shown)
                    plt.subplots_adjust(wspace=0.15, hspace=0.4)
                    plt.axis('off')
                    
                    img = Image.open(os.path.join(image_dir, str(filename)))
                    plt.imshow(img)
                    
                    plt.title(f'{idx+1}', fontsize=6)
                    
                    if (idx+1 - total_images_shown) >= num_images or (idx+1) == total_images:
                        if save_filename:
                            pdf.savefig()
                        if show:
                            plt.show()
                        if (idx + 1) < total_images:
                            plt.figure(figsize=(num_columns, num_rows+1))
                            plt.axis('off')
                            plt.suptitle(f'CLUSTER {clust_idx + 1}', y=0.91, fontsize=14)
                            total_images_shown += num_images
                        else:
                            break    


############################################################################################################
# PLOTTING PROJECTIONS
############################################################################################################

def _ray_to_rect_boundary(p, d, xlim, ylim):
    """
    Intersect ray p + t d (t>0) with rectangle [xlim]x[ylim].
    Returns (xi, yi, edge), where edge in {'left','right','bottom','top'}.
    """
    px, py = float(p[0]), float(p[1])
    dx, dy = float(d[0]), float(d[1])
    eps = 1e-12

    t_candidates = []
    edges = []

    # x = xlim[1] (right)
    if dx > eps:
        t = (xlim[1] - px) / dx
        if t > 0:
            t_candidates.append(t); edges.append('right')
    # x = xlim[0] (left)
    if dx < -eps:
        t = (xlim[0] - px) / dx
        if t > 0:
            t_candidates.append(t); edges.append('left')
    # y = ylim[1] (top)
    if dy > eps:
        t = (ylim[1] - py) / dy
        if t > 0:
            t_candidates.append(t); edges.append('top')
    # y = ylim[0] (bottom)
    if dy < -eps:
        t = (ylim[0] - py) / dy
        if t > 0:
            t_candidates.append(t); edges.append('bottom')

    if not t_candidates:
        # Degenerate: aim outward by whichever axis has nonzero component
        # This is rare; just fall back to right edge.
        t = (xlim[1] - px) / (dx if abs(dx) > eps else (np.sign(dx)+eps))
        xi, yi, edge = px + t * dx, py + t * dy, 'right'
        return xi, yi, edge

    tmin_idx = int(np.argmin(t_candidates))
    tmin = t_candidates[tmin_idx]
    edge = edges[tmin_idx]
    xi, yi = px + tmin * dx, py + tmin * dy
    return xi, yi, edge


def _space_along_edge(anchor_vals, min_gap, lo, hi):
    """
    1D greedy spacing to avoid overlap along an edge.
    anchor_vals: desired positions (x for top/bottom, y for left/right), sorted.
    min_gap: minimum spacing in data units.
    Returns equally-long array of spaced positions.
    """
    if not anchor_vals:
        return []

    spaced = [anchor_vals[0]]
    for v in anchor_vals[1:]:
        spaced.append(max(v, spaced[-1] + min_gap))

    # If we overflow, shift back to fit within [lo, hi]
    overflow = spaced[-1] - hi
    if overflow > 0:
        spaced = [s - overflow for s in spaced]
        # If we underflow at the start, clamp to lo with min_gap spacing forward
        under = lo - spaced[0]
        if under > 0:
            spaced = [lo + i*min_gap for i in range(len(spaced))]

    return spaced


def print_projection(clusters, 
                     model_dict, 
                     projection, 
                     show_images=False, 
                     with_filenames=False,
                     cleaned=False, 
                     negative_silhouette=False, 
                     clusters_per_marker=None,
                     centroid_label=False,
                     image_labels=None,
                     cluster_descriptions=None,
                     zoom=1,
                     outside_margin=0.5,
                     gap_frac=0.02):

    clusters = np.array(clusters)
    print("# Vectors: ", len(clusters))
    unique_labels = np.unique(clusters)

    if len(unique_labels) > 40:
        print_labels = False
    else:
        print_labels = True

    # Create a color map
    color_map = glasbey.create_palette(palette_size=len(unique_labels), chroma_bounds=(60, 100), lightness_bounds=(30, 80))

    if show_images or (cluster_descriptions is not None):

        IMAGE_SIZE = (50, 50)  # Resize images for plotting
        IMAGE_FOLDER = model_dict['image_dir']
        SPRITE_FILE = model_dict['sprite']
        filenames = model_dict['filenames']
        NUM_IMAGES = len(filenames)
        if cleaned:
            filenames_inliers = model_dict['filenames_inliers']

        indices = np.arange(NUM_IMAGES)
        
        # Load the sprite image
        sprite_image = Image.open(SPRITE_FILE)
        sprite_width, sprite_height = sprite_image.size
        grid_size = int(np.ceil(np.sqrt(NUM_IMAGES)))

        def get_sprite_image(sprite_image, index, image_size, grid_size):
            row = index // grid_size
            col = index % grid_size
            left = col * image_size[0]
            upper = row * image_size[1]
            right = left + image_size[0]
            lower = upper + image_size[1]
            # Ensure the sprite extraction doesn't go out of bounds
            if right > sprite_width or lower > sprite_height:
                return None
            return sprite_image.crop((left, upper, right, lower))
    
    if show_images:
        
        fig, ax = plt.subplots(figsize=(40, 40))
        
        text_labels = []
        # Plot the data points with unique colors for each cluster

        colors = {}

        for j, label in enumerate(unique_labels):

            if label == -1:
                colors[label] = '#dbdbdb'
            else:
                colors[label] = color_map[j]

        for i, (x, y) in enumerate(projection):
                
            img_index = indices[i]
            img_index = filenames.index(filenames_inliers[img_index]) if cleaned else img_index
            img = get_sprite_image(sprite_image, img_index, IMAGE_SIZE, grid_size)
            if img is not None:
                img = np.array(img)
                
                im = OffsetImage(img, zoom=zoom)  # Adjust zoom if needed

                ab = AnnotationBbox(im, (x, y), frameon=True, bboxprops=dict(edgecolor=colors[clusters[i]],linewidth=6*zoom), pad=0)
                ax.add_artist(ab)

                if with_filenames:
                    ax.text(x, y - 5 * zoom, filenames[img_index], fontsize=8, ha='center', va='top', color='black')

        if print_labels:
            
            for j, label in enumerate(unique_labels):

                if label == -1:
                    color = '#dbdbdb'
                else:
                    color = color_map[j]
                    
                mask = np.array(clusters) == label

                # Calculate the centroid of each cluster
                centroid = np.mean(projection[mask], axis=0)

                # Add cluster label near the centroid
                text_labels.append(ax.text(centroid[0], centroid[1], f'Cluster {label}', fontsize=12, fontweight='bold',
                        ha='center', va='center', bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3')))
                
                # Adjust the positions of the text labels to avoid overlaps
                #adjust_text(text_labels, projection[:, 0], projection[:, 1], ax=ax, arrowprops=dict(arrowstyle='->', color='black'))

        # Set plot limits and labels
        ax.set_xlim(projection[:, 0].min()-1, projection[:, 0].max()+1)
        ax.set_ylim(projection[:, 1].min()-1, projection[:, 1].max()+1)
        ax.set_title("2D Embedding with Images")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.axis('off')
        
        plt.tight_layout()
        plt.show()
                
    if show_images==False:
        
        text_labels = []
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot the data points with unique colors for each cluster
        for i, label in enumerate(unique_labels):

            if label == -1:
                color = '#dbdbdb'
                shape = 'o'
            else:
                color = color_map[i]
                shape = 'o'
                if clusters_per_marker:
                    if (i // clusters_per_marker == 0):
                        shape = 'o'
                    if (i // clusters_per_marker == 1):
                        shape = '^'
                    if (i // clusters_per_marker == 2):
                        shape = 's'
                    if (i// clusters_per_marker == 3):
                        shape = 'd'    
                    if (i // clusters_per_marker == 4):
                        shape = 'P'
                    if (i // clusters_per_marker == 5):
                        shape = '*'
                    if (i// clusters_per_marker >= 6):
                        shape = 'X'
            
            mask = np.array(clusters) == label
                    
            ax.scatter(projection[mask, 0], projection[mask, 1], color=color, label=f'Cluster {label}', marker=shape, edgecolor=color, alpha=0.5, s=15)
            ax.set_title("2D Embedding with Images")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.axis('off')
            
            if print_labels:
                
                if centroid_label:

                    # Calculate the centroid of each cluster
                    centroid = np.mean(projection[mask], axis=0)

                    # Add cluster label near the centroid
                    text_labels.append(ax.text(centroid[0], centroid[1], f'Cluster {label}', fontsize=14, fontweight='bold',
                            ha='center', va='center', bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3')))
                    
                elif cluster_descriptions is not None:
                    
                    #indices = np.arange(NUM_IMAGES)

                    if image_labels is not None and (label != -1):
                        if image_labels == "random":
                            # Get the indices where mask equals 1 and choose a random one
                            indices = np.where(mask == 1)[0]
                            if indices.size > 0:
                                chosen_index = np.random.choice(indices)
                                image_idx = chosen_index

                        print("LABEL", label)
                        
                        # Find the (x, y) location of the image in the projection
                        x, y = projection[image_idx]

                        ax.plot(x, y, 'o', markersize=12, markeredgecolor='black', markerfacecolor=color)
                        
                        # Compute direction to move label outward (from centroid to point, then extend)
                        cluster_points = projection[clusters == label]
                        centroid = np.mean(cluster_points, axis=0)
                        direction = np.array([x, y]) - centroid
                        direction = direction / (np.linalg.norm(direction) + 1e-8)
                        # Place label outside the convex hull of the cluster
                        label_offset = 2.0  # adjust as needed
                        label_pos = np.array([x, y]) + direction * label_offset
            
                        # Draw a line from image to label
                        ax.plot([x, label_pos[0]], [y, label_pos[1]], color='black', linewidth=2)
            
                        # Add text label (description or fallback)
                        if cluster_descriptions is not None:
                            label_text = cluster_descriptions.loc[cluster_descriptions['cluster'] == label, 'description'].iloc[0]
                        else:
                            label_text = f"Cluster {label}"
                        ax.text(label_pos[0], label_pos[1], label_text, fontsize=10,
                                ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))

                        # Display the image below the text label
                        img = get_sprite_image(sprite_image, image_idx, IMAGE_SIZE, grid_size)
                        if img is not None:
                            img = np.array(img)
                            im = OffsetImage(img, zoom=zoom)
                            ab = AnnotationBbox(im, (label_pos[0], label_pos[1] - 1.5 * zoom), frameon=True, pad=0.2)
                            ax.add_artist(ab)

        plt.tight_layout()
        plt.show()


import seaborn as sns
from numpy.linalg import svd as _svd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker

from scipy.spatial import ConvexHull

def _unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _build_convex_hull_polygon(X):
    """Return hull vertices (CCW) as an (M,2) array, closed (last == first)."""
    hull = ConvexHull(X)
    poly = X[hull.vertices]
    # ensure closed for easy edge iteration
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly  # CCW by scipy convention

def _ray_to_polygon_boundary(p, d, poly):
    """
    Intersect ray p + t d (t>=0) with polygon edges.
    Returns: (xi, yi), edge_index, outward_normal (unit).
    """
    d = _unit(d)
    best_t = np.inf
    hit_pt = None
    hit_i = -1

    for i in range(len(poly)-1):
        a = poly[i]
        b = poly[i+1]
        e = b - a  # edge vector

        # Solve p + t d = a + u e  with t>=0 and 0<=u<=1
        # Stack as [d, -e] * [t, u]^T = (a - p)
        M = np.array([d, -e]).T  # 2x2
        rhs = a - p
        det = np.linalg.det(M)
        if abs(det) < 1e-12:
            continue
        t, u = np.linalg.solve(M, rhs)
        if t >= 0 and 0.0 <= u <= 1.0:
            if t < best_t:
                best_t = t
                hit_pt = p + t * d
                hit_i = i

    if hit_pt is None:
        # Fallback: no intersection found (degenerate cases) -> project to nearest hull vertex
        i_min = np.argmin(np.linalg.norm(poly[:-1] - p, axis=1))
        hit_pt = poly[i_min]
        hit_i = i_min

    # Outward normal for CCW polygon = rotate edge by +90°: n = (ey, -ex)
    a = poly[hit_i]
    b = poly[hit_i + 1]
    e = b - a
    n = _unit(np.array([e[1], -e[0]]))  # outward normal for CCW
    return hit_pt[0], hit_pt[1], hit_i, n

def _hull_segment_lengths(poly):
    """Return segment lengths and cumulative arc-length along closed hull polygon."""
    segs = poly[1:] - poly[:-1]     # (M-1, 2)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    return lens, cum  # lens has length M-1, cum has length M

def _point_to_hull_arclength(poly, cum, i, u):
    """
    Map a point on edge i at fraction u∈[0,1] to absolute arclength s∈[0, L).
    poly is closed; cum is cumulative lengths with len(poly) == len(cum).
    """
    return float(cum[i] + u * (cum[i+1] - cum[i]))

def _space_on_circle(values, min_gap, circumference):
    """
    Enforce cyclic min spacing on [0, circumference).
    Greedy pass: sort, then push right; final wrap-around fix.
    """
    if not values:
        return []
    L = float(circumference)
    vals = np.sort(np.array(values, dtype=float))
    out = vals.copy()

    # forward pass
    for k in range(1, len(out)):
        if out[k] - out[k-1] < min_gap:
            out[k] = out[k-1] + min_gap

    # wrap-around: if last overlaps first across the boundary, shift all by delta
    if L - out[-1] + out[0] < min_gap:
        delta = min_gap - (L - out[-1] + out[0])
        out += delta
    out %= L

    # Optional second forward pass to clean up after wrap
    out.sort()
    for k in range(1, len(out)):
        if out[k] - out[k-1] < min_gap:
            out[k] = out[k-1] + min_gap
    out %= L

    # Preserve original order mapping (stable assignment by nearest)
    # (Keeps labels attached to their clusters consistently)
    assign = []
    for v in values:
        j = int(np.argmin(np.abs(out - v)))
        assign.append(out[j])
        out = np.delete(out, j)
    return assign

def _arclength_to_point(poly, cum, s):
    """
    Map absolute arclength s back to a point on the hull and its edge index + outward normal.
    """
    L = cum[-1]
    s = float(s % L)
    i = int(np.searchsorted(cum, s, side='right') - 1)
    i = max(0, min(i, len(cum) - 2))
    a = poly[i]
    b = poly[i+1]
    seg_len = cum[i+1] - cum[i]
    u = 0.0 if seg_len <= 1e-12 else (s - cum[i]) / seg_len
    p = a + u * (b - a)
    e = b - a
    n = _unit(np.array([e[1], -e[0]]))  # outward normal (CCW)
    return p, i, n


def sprint_projection(clusters,
                     model_dict, 
                     projection, 
                     show_images=False, 
                     clusters_per_marker=None,
                     centroid_label=False,
                     image_labels=None,
                     cluster_descriptions=None,
                     zoom=1,
                     outside_margin=0.7,
                     gap_frac=0.2,
                     probability_desaturate=False,
                     hdbscan_model=None):

    clusters = np.array(clusters)
    print("# Vectors: ", len(clusters))
    unique_labels = np.unique(clusters)

    if len(unique_labels) > 40:
        print_labels = False
    else:
        print_labels = True

    # Create a color map
    color_map = glasbey.create_palette(palette_size=len(unique_labels), chroma_bounds=(60, 100), lightness_bounds=(30, 80))

    if show_images or (cluster_descriptions is not None):

        IMAGE_SIZE = (50, 50)  # Resize images for plotting
        IMAGE_FOLDER = model_dict['image_dir']
        SPRITE_FILE = model_dict['sprite']
        filenames = model_dict['filenames']
        NUM_IMAGES = len(filenames)
        
        # Load the sprite image
        sprite_image = Image.open(SPRITE_FILE)
        sprite_width, sprite_height = sprite_image.size
        grid_size = int(np.ceil(np.sqrt(NUM_IMAGES)))

        def get_sprite_image(sprite_image, index, image_size, grid_size):
            row = index // grid_size
            col = index % grid_size
            left = col * image_size[0]
            upper = row * image_size[1]
            right = left + image_size[0]
            lower = upper + image_size[1]
            # Ensure the sprite extraction doesn't go out of bounds
            if right > sprite_width or lower > sprite_height:
                return None
            return sprite_image.crop((left, upper, right, lower))
                
    if show_images==False:
        
        def hex_to_rgb(hex):
            if hex.startswith('#'):
                hex = hex[1:]
            return tuple(int(hex[i:i+2], 16)/255 for i in (0, 2, 4))
        
        text_labels = []
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15, 15))

        proj = np.asarray(projection)
        hull_poly = _build_convex_hull_polygon(proj)                # closed CCW polygon
        lens, cum = _hull_segment_lengths(hull_poly)                # seg lengths + cumulative
        perimeter = float(cum[-1])
        dataset_centroid = proj.mean(axis=0)
        label_items = []  # collect items for all labels (we'll add arclength 's' too)

        # Plot the data points with unique colors for each cluster
        for i, label in enumerate(unique_labels):

            if label == -1:
                color = '#dbdbdb'
                shape = 'o'
            else:
                color = color_map[i]
                shape = 'o'
                if clusters_per_marker:
                    if (i // clusters_per_marker == 0):
                        shape = 'o'
                    if (i // clusters_per_marker == 1):
                        shape = '^'
                    if (i // clusters_per_marker == 2):
                        shape = 's'
                    if (i// clusters_per_marker == 3):
                        shape = 'd'    
                    if (i // clusters_per_marker == 4):
                        shape = 'P'
                    if (i // clusters_per_marker == 5):
                        shape = '*'
                    if (i// clusters_per_marker >= 6):
                        shape = 'X'
            
            point_color = color
            
            mask = np.array(clusters) == label
            
            if probability_desaturate and (hdbscan_model is not None) and (label != -1):

                cluster_member_colors = [hex_to_rgb(color) for i in mask]
                
                point_color = []
                
                for x, p in zip(cluster_member_colors, hdbscan_model.probabilities_):
                    point_color.append(sns.desaturate(x, p))

                point_color = np.array(point_color)
                point_color = point_color[mask]

            ax.scatter(projection[mask, 0], projection[mask, 1], color=point_color, label=f'Cluster {label}', marker=shape, edgecolor="none", alpha=0.5, s=15)
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            
            if print_labels:
                
                if centroid_label:

                    # Calculate the centroid of each cluster
                    centroid = np.mean(projection[mask], axis=0)

                    # Add cluster label near the centroid
                    text_labels.append(ax.text(centroid[0], centroid[1], f'Cluster {label}', fontsize=12, fontweight='bold',
                            ha='center', va='center', bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3')))
                    
                elif cluster_descriptions is not None:

                    if (image_labels is not None) and (label != -1 and label != 0):
                        if image_labels == "random":
                            # Get the indices where mask is True and choose a random one   # CHANGED: ==1 -> just mask
                            idxs = np.where(mask)[0]
                            if idxs.size > 0:
                                chosen_index = np.random.choice(idxs)
                                image_idx = chosen_index
                            else:
                                continue
                        elif image_labels == "selected":
                            filename = cluster_descriptions.loc[cluster_descriptions['cluster'] == label, 'image 1'].iloc[0]
                            image_idx = filenames.index(filename)

                        # Find the (x, y) location of the image in the projection
                        x, y = projection[image_idx]

                        # Optional: highlight chosen point (kept from your code)
                        ax.plot(x, y, 'o', markersize=4, markeredgecolor='black', markerfacecolor=color)

                        # Compute direction centroid -> point
                        centroid = np.mean(projection, axis=0)
                        direction = np.array([x, y]) - centroid
                        direction = direction / (np.linalg.norm(direction) + 1e-8)

                        # --- NEW: intersect ray with bounding rectangle, then push just outside ---
                        # Intersect ray with convex hull boundary, then nudge outward
                        # Intersect ray with convex hull boundary
                        xi, yi, edge_idx, outward_n = _ray_to_polygon_boundary(np.array([x, y]),
                                                                            direction,
                                                                            hull_poly)

                        # Compute precise fraction u along the hit edge for arclength bookkeeping
                        a = hull_poly[edge_idx]
                        b = hull_poly[edge_idx + 1]
                        edge_vec = b - a
                        edge_len = np.linalg.norm(edge_vec)
                        if edge_len < 1e-12:
                            u = 0.0
                        else:
                            u = np.clip(np.dot(np.array([xi, yi]) - a, edge_vec) / (edge_len**2), 0.0, 1.0)

                        s_abs = _point_to_hull_arclength(hull_poly, cum, edge_idx, u)

                        if cluster_descriptions is not None:
                            label_text = f"{label}. " + cluster_descriptions.loc[
                                cluster_descriptions['cluster'] == label, 'description'
                            ].iloc[0].replace(',', ',\n')
                        else:
                            label_text = f"Cluster {label}"

                        label_items.append({
                            'lbl': label,
                            'pt': (float(x), float(y)),
                            'edge_idx': int(edge_idx),
                            's': float(s_abs),              # <-- perimeter coordinate
                            'normal': outward_n.astype(float),
                            'text': label_text,
                            'image_idx': int(image_idx)
                        })


        if print_labels and (cluster_descriptions is not None) and label_items:
            # --- Hard spacing along the hull perimeter ---
            min_gap = float(gap_frac) * perimeter  # e.g., 0.2 => 20% of perimeter between anchors
            s_vals = [it['s'] for it in label_items]
            s_spaced = _space_on_circle(s_vals, min_gap, perimeter)

            # Update anchors from spaced arclengths
            anchors = []
            for it, s_new in zip(label_items, s_spaced):
                p_on_hull, edge_i, n_out = _arclength_to_point(hull_poly, cum, s_new)
                ax_anch = (float(p_on_hull[0] + n_out[0]*outside_margin),
                        float(p_on_hull[1] + n_out[1]*outside_margin))
                anchors.append(ax_anch)
                it['anchor'] = ax_anch  # stash for drawing

            # --- Create invisible proxies and let adjust_text do micro-nudges ---
            temp_texts, finals_xy = [], []
            for it in label_items:
                lx, ly = it['anchor']
                t = ax.text(lx, ly, it['text'], fontsize=16, alpha=0.0, weight="light")
                temp_texts.append(t)
                finals_xy.append((it, lx, ly))

            if temp_texts:
                # crank up forces if you still want more air between boxes
                adjust_text(
                    temp_texts,
                    expand_points=(1.15, 1.2),
                    expand_text=(1.2, 1.25),
                    force_text=(0.6, 0.6),
                    lim=200,
                    only_move={'text': 'xy'}
                )

            _PULL_IN = 0.15  # optional subtle pull-back toward anchors

            # --- Draw the ABs at adjusted positions ---
            for (it, lx0, ly0), t in zip(finals_xy, temp_texts if temp_texts else [None]*len(finals_xy)):
                if t is not None:
                    lx_adj, ly_adj = t.get_position()
                else:
                    lx_adj, ly_adj = lx0, ly0
                lx = lx0 + _PULL_IN * (lx_adj - lx0)
                ly = ly0 + _PULL_IN * (ly_adj - ly0)

                img = get_sprite_image(sprite_image, it['image_idx'], IMAGE_SIZE, grid_size)
                if img is None:
                    continue
                im = OffsetImage(np.array(img), zoom=zoom)
                textarea = TextArea(it['text'], textprops=dict(fontsize=12))
                packed = VPacker(children=[textarea, im], align="center", pad=0, sep=2)

                ab = AnnotationBbox(
                    packed,
                    xy=it['pt'],
                    xybox=(lx, ly),
                    xycoords='data',
                    boxcoords='data',
                    frameon=False,
                    bboxprops=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='black'),
                    arrowprops=dict(arrowstyle='-', lw=1, color='black')
                )
                ax.add_artist(ab)

            for t in temp_texts:
                t.remove()

        plt.tight_layout()
        plt.show()

import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker

# NOTE: All hull/spacing helpers removed.

def _hex_to_rgb01(hex_color):
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

def _coerce_xy(val):
    """
    Accepts (x, y), [x, y], np.array([x, y]), or a string like 'x,y' or '(x, y)'.
    Returns (float(x), float(y)).
    """
    if isinstance(val, (tuple, list, np.ndarray)) and len(val) == 2:
        return float(val[0]), float(val[1])
    if isinstance(val, str):
        s = val.strip().strip('()[]')
        parts = s.split(',')
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    raise ValueError(f"Cannot parse coordinates from: {val!r}")

def wrap_prefer_breaks(s: str, bad_map={' ': 0, ',': 1, '/': 1}, max_width=10) -> str:
    lines = []
    line_start = 0
    last_break = None       # index of last seen break char
    last_break_keep = 0     # how many chars to include from the break

    for i, ch in enumerate(s):
        # track preferred breakpoints
        if ch in bad_map:
            last_break = i
            last_break_keep = bad_map[ch]

        # if we've exceeded width, break
        if i - line_start + 1 > max_width:
            if last_break is not None and last_break >= line_start:
                cut = last_break + last_break_keep
                lines.append(s[line_start:cut])
                line_start = last_break + 1
                last_break = None
                last_break_keep = 0
            else:
                # no good breakpoint in window: hard break before current char
                lines.append(s[line_start:i])
                line_start = i

    # add the remainder
    if line_start < len(s):
        lines.append(s[line_start:])

    return "\n".join(lines)

def pprint_projection(clusters,
                      model_dict, 
                      projection, 
                      show_images=False, 
                      clusters_per_marker=None,
                      image_labels=None,
                      cluster_descriptions=None,
                      zoom=1,
                      probability_desaturate=False,
                      hdbscan_model=None):

    clusters = np.array(clusters)
    print("# Vectors: ", len(clusters))
    unique_labels = np.unique(clusters)

    # color map
    color_map = glasbey.create_palette(
        palette_size=len(unique_labels),
        chroma_bounds=(60, 100),
        lightness_bounds=(30, 80)
    )

    # Optional image setup
    if show_images or (cluster_descriptions is not None):
        IMAGE_SIZE = (400, 400)
        filenames = model_dict['filenames']
        image_dir = model_dict['image_dir']
        NUM_IMAGES = len(filenames)

        def load_image_from_file(filename, image_dir, image_size):
            """Load and resize an image directly from file."""
            import cv2
            filepath = os.path.join(image_dir, str(filename))
            if not os.path.exists(filepath):
                return None
            img = cv2.imread(filepath)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            return Image.fromarray(img)

    # figure
    fig, ax = plt.subplots(figsize=(15, 15), dpi=300)

    proj = np.asarray(projection)

    # plot points
    for i, label in enumerate(unique_labels):
        if label == -1:
            color = '#dbdbdb'
            shape = 'o'
        else:
            color = color_map[i]
            shape = 'o'
            if clusters_per_marker:
                if (i // clusters_per_marker == 0): shape = 'o'
                elif (i // clusters_per_marker == 1): shape = '^'
                elif (i // clusters_per_marker == 2): shape = 's'
                elif (i // clusters_per_marker == 3): shape = 'd'
                elif (i // clusters_per_marker == 4): shape = 'P'
                elif (i // clusters_per_marker == 5): shape = '*'
                else: shape = 'X'

        mask = (clusters == label)

        # probability-based desaturation (optional)
        if probability_desaturate and (hdbscan_model is not None) and (label != -1):
            base_rgb = _hex_to_rgb01(color)
            # replicate base color for all points in cluster
            cluster_member_colors = [base_rgb] * mask.sum()
            # filter probabilities to current mask
            probs = np.asarray(hdbscan_model.probabilities_)[mask]
            # desaturate each by its probability
            point_color = [sns.desaturate(base_rgb, p) for p in probs]
        else:
            point_color = color

        ax.scatter(proj[mask, 0], proj[mask, 1],
                   color=point_color, label=f'Cluster {label}',
                   marker=shape, edgecolor="none", alpha=0.5, s=15)

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.axis('off')

    # === MANUAL LABEL/IMAGE PLACEMENT =======================================
    # If you pass image_labels=="selected" (or "random") AND cluster_descriptions with:
    #   - a row for each cluster label
    #   - column 'image 1' giving the filename to pull from the sprite
    #   - column 'coordinates' giving (x, y) for where to place the box
    # we will drop the box at exactly those coordinates.
    if (cluster_descriptions is not None) and (image_labels is not None):
        for i, label in enumerate(unique_labels):
            if label in (-1, 0):  # match your previous exclusions
                continue

            # pick image index
            if image_labels == "random":
                idxs = np.where(clusters == label)[0]
                if idxs.size == 0:
                    continue
                image_idx = int(np.random.choice(idxs))
            elif image_labels == "selected":
                # lookup filename from table, then map to filenames list
                filename = cluster_descriptions.loc[
                    cluster_descriptions['cluster'] == label, 'image 1'
                ].iloc[0]
                image_idx = filenames.index(filename)
            else:
                continue  # unknown mode -> skip

            # tail point (data location of chosen image)
            tail_x, tail_y = proj[image_idx]

            # head/box location (your manual coordinates)
            coord_val = cluster_descriptions.loc[
                cluster_descriptions['cluster'] == label, 'coordinates'
            ].iloc[0]
            box_x, box_y = _coerce_xy(coord_val)

            # optional highlight of the chosen point
            ax.plot(tail_x, tail_y, 'o', markersize=4, markeredgecolor='black',
                    markerfacecolor=color_map[i] if label != -1 else '#dbdbdb')

            # load image directly from file
            img = load_image_from_file(filenames[image_idx], image_dir, IMAGE_SIZE)
            if img is None:
                continue

            im = OffsetImage(np.array(img), zoom=zoom)

            if 'description' in cluster_descriptions.columns:
                
                description = cluster_descriptions.loc[
                                  cluster_descriptions['cluster'] == label, 'description'
                              ].iloc[0]

                description_new = wrap_prefer_breaks(description, max_width=15)
                label_text = f"{label}\n" + description_new
                textarea = TextArea(label_text, textprops=dict(fontsize=20, horizontalalignment="center", fontfamily="arial"))
                packed = VPacker(children=[textarea, im], align="center", pad=0, sep=5)
            else:
                packed = im  # just the image

            ab = AnnotationBbox(
                packed,
                xy=(tail_x, tail_y),      # arrow tail at data point
                xybox=(box_x, box_y),     # your manual coordinates
                xycoords='data',
                boxcoords='data',
                frameon=False,
                bboxprops=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='black'),
                arrowprops=dict(arrowstyle='-', lw=1, color='black')
            )
            ax.add_artist(ab)
    # ========================================================================

    plt.tight_layout()
    plt.show()
