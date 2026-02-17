from email import utils
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import glasbey
import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap

from PIL import Image
from adjustText import adjust_text
from contextlib import ExitStack
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.base import clone
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker
from tqdm.notebook import tqdm
from time import time
from collections import defaultdict, Counter
from PIL import Image

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

def print_cluster_pdf(clustering, 
                        vectors,
                        model_dict,
                        sort="random",
                        print_filename=False,
                        print_distance=False,
                        save_filename="cluster_images.pdf",
                        show=False):

    # Get cluster labels and compute distances to centroids
    clusters_ = clustering.labels_
    filenames = model_dict['filenames']
    image_dir = model_dict['image_dir']
    
    # Group filenames and distances by cluster
    clustered_data = defaultdict(list)
    
    if sort == "random":
        rng = np.random.default_rng(seed=42)
        rand_nums = rng.random(len(filenames))
        
        for filename, rand_num, label in zip(filenames, rand_nums, clusters_):
            clustered_data[label].append((rand_num, filename))
        
    if sort == "centroid":
        centroids = clustering.cluster_centers_
        distances_to_centroid = np.linalg.norm(vectors - centroids[clusters_], axis=1)
        
        for filename, distance, label in zip(filenames, distances_to_centroid, clusters_):
            clustered_data[label].append((distance, filename))

    # Sort clusters by the number of elements and sort items within each cluster by distance/rand_num
    sorted_clustered_data = {
        label: sorted(items, key=lambda x: x[0])  # Sort items in each cluster by distance/rand_num
        for label, items in sorted(clustered_data.items(), key=lambda x: len(x[1]))  # Sort clusters by size
    }

    # Extract sorted filenames and distances
    clustered_filenames = {label: [filename for _, filename in items] for label, items in sorted_clustered_data.items()}
    
    if sort == "centroid":
        clustered_distances = {label: [distance for distance, _ in items] for label, items in sorted_clustered_data.items()}

    with PdfPages(save_filename) as pdf:
        
        cluster_order = sorted([k for k in clustered_filenames.keys() if k != -1]) + ([-1] if -1 in clustered_filenames else [])
        
        for clust_idx, cluster in enumerate(cluster_order):
            images = clustered_filenames[cluster]
    
            print(f'CLUSTER: {cluster}')
            print(f'N: {len(images)}')
            print('-'*50)
            
            total_images = len(images)
            total_images_shown = 0
            num_images_per_page = 120

            num_columns = 10
            num_rows = (num_images_per_page - 1) // num_columns + 1
            
            page_idx = 1
            plt.figure(figsize=(num_columns, num_rows+1))
            plt.axis('off')
            plt.suptitle(f'CLUSTER {cluster}', y=0.91, fontsize=12)

            for idx, filename in enumerate(images):
                
                    if sort == "centroid":
                        distance = np.round(clustered_distances[cluster][idx], 4)

                    plt.subplot(num_rows, num_columns, idx+1-total_images_shown)
                    plt.subplots_adjust(wspace=0.15, hspace=0.4)
                    plt.axis('off')
                    
                    img = Image.open(os.path.join(image_dir, str(filename)))
                    plt.imshow(img)
                    
                    title = f"{idx+1}"
                    
                    if print_filename:
                            title += f": {filename}"
                    if print_distance:
                            title += f"\n{distance}"
                            
                    plt.title(title, fontsize=6)
                    
                    if (idx+1 - total_images_shown) >= num_images_per_page or (idx+1) == total_images:
                        if save_filename:
                            pdf.savefig()
                        if show:
                            plt.show()
                        if (idx + 1) < total_images:
                            plt.figure(figsize=(num_columns, num_rows+1))
                            plt.axis('off')
                            plt.suptitle(f'CLUSTER {cluster}', y=0.91, fontsize=14)
                            total_images_shown += num_images_per_page
                        else:
                            break    


############################################################################################################
# PLOTTING PROJECTIONS
############################################################################################################

def plot_umap(vectors,
              model_dict,
              plot2d = True,
              show_images=False,
              zoom=0.3,
              n_neighbors=50,
              min_dist=0.0,
              metric="cosine",
              clusters=None,
              figsize=(30,30),
              subsample_size=5000
              ):

    umap_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    X_umap_2d = umap_2d.fit_transform(np.array(vectors))

    if plot2d:
        print(f"Trustworthiness: {trustworthiness(vectors, X_umap_2d, n_neighbors=30):.2f}")

        if show_images:
            plot_with_images(X_umap_2d, model_dict, figsize=figsize, zoom=zoom, cleaned=False, subsample_size=subsample_size)

        else:
            plt.figure(figsize=(20, 20))
            plt.scatter(X_umap_2d[:, 0], X_umap_2d[:, 1])
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.title('UMAP Visualization')
            plt.show()

    return X_umap_2d

def print_projection(clusters, 
                     model_dict, 
                     projection, 
                     show_images=False, 
                     with_filenames=False,
                     cleaned=False,
                     clusters_per_marker=None,
                     centroid_label=False,
                     image_labels=None,
                     cluster_descriptions=None,
                     zoom=1):

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

def print_projection_selected_images(clusters,
                                    model_dict, 
                                    projection, 
                                    show_images=False, 
                                    image_labels=None,
                                    cluster_descriptions=None,
                                    zoom=1,
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

        mask = (clusters == label)

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
    #   - column 'selected_image' giving the filename to pull from the sprite
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
                    cluster_descriptions['cluster'] == label, 'selected_image'
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