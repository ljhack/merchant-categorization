'''
Clustering algorithms and helpers
'''

## Clustering dependencies
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN

## Dimensionality reduction dependencies
from sklearn.decomposition import PCA
import umap.umap_ as umap
# from sklearn.metrics.pairwise import cosine_distances

## Visualization dependencies
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

## Other helpers
from collections import Counter
import numpy as np
from datetime import datetime

## Configurations
import configs as CFG

CURRENT_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M')

def run_pca(**kwargs):
    '''
    Runs PCA on embeddings
    '''
    scaled_embeddings = kwargs.get('scaled_embeddings')
    n_components = kwargs.get('n_components', 10)
    random_state = kwargs.get('random_state', 42)
    visualize = kwargs.get('visualize', False)

    print('Running PCA...')
    print(f'Number of components: {n_components}')

    ## Run PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_embeddings = pca.fit_transform(scaled_embeddings)

    if visualize:
        ## Visualize explained variance ratio
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('Explained variance ratio')
        plt.savefig(f'{CFG.REPORTS_PATH}pca_explained_variance_ratio_{CURRENT_TIME}.png')

        ## Visualize PCA components
        plt.figure(figsize=(10, 5))
        sns.heatmap(pca.components_, cmap='viridis')
        plt.xlabel('Features')
        plt.ylabel('Principal components')
        plt.title('PCA components')
        plt.savefig(f'{CFG.REPORTS_PATH}pca_components_{CURRENT_TIME}.png')

        ## Visualize PCA components as word clouds
        for i in range(n_components):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join([str(x) for x in pca.components_[i]]))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'PCA component {i+1}')
            plt.savefig(f'{CFG.REPORTS_PATH}pca_component_{i+1}_{CURRENT_TIME}.png')
                
        ## Visualize PCA components as bar plots
        for i in range(n_components):
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(pca.components_[i])), pca.components_[i])
            plt.xlabel('Features')
            plt.ylabel('Value')
            plt.title(f'PCA component {i+1}')
            plt.savefig(f'{CFG.REPORTS_PATH}pca_component_{i+1}_bar_{CURRENT_TIME}.png')

        ## Visualize in 3d
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], pca_embeddings[:, 2])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('PCA 3D')
        plt.savefig(f'{CFG.REPORTS_PATH}pca_3d_{CURRENT_TIME}.png')
        plt.close()

    return pca_embeddings



def run_umap(**kwargs):
    '''
    Runs UMAP on embeddings
    '''
    scaled_embeddings = kwargs.get('scaled_embeddings')
    n_components = kwargs.get('n_components', 10)
    n_neighbors = kwargs.get('n_neighbors', 15)
    min_dist = kwargs.get('min_dist', 0.1)
    metric = kwargs.get('metric', 'cosine')
    # random_state = kwargs.get('random_state', 42)
    visualize = kwargs.get('visualize', False)

    print(f'Running UMAP with n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}')

    ## Run UMAP
    umap_embeddings = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        # min_dist=min_dist,
        metric=metric,
        # random_state=random_state
    ).fit_transform(scaled_embeddings)

    if visualize:
        ## Visualize UMAP embeddings
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1])
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('UMAP')
        plt.savefig(f'{CFG.REPORTS_PATH}umap_{CURRENT_TIME}.png')

        # ## Visualize UMAP embeddings as word clouds
        # for i in range(n_components):
        #     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join([str(x) for x in umap_embeddings[:, i]]))
        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(wordcloud, interpolation='bilinear')
        #     plt.axis('off')
        #     plt.title(f'UMAP component {i+1}')
        #     plt.savefig(f'{CFG.REPORTS_PATH}umap_component_{i+1}_{CURRENT_TIME}.png')

        # ## Visualize UMAP embeddings as bar plots
        # for i in range(n_components):
        #     plt.figure(figsize=(10, 5))
        #     plt.bar(range(len(umap_embeddings[:, i])), umap_embeddings[:, i])
        #     plt.xlabel('Features')
        #     plt.ylabel('Value')
        #     plt.title(f'UMAP component {i+1}')
        #     plt.savefig(f'{CFG.REPORTS_PATH}umap_component_{i+1}_bar_{CURRENT_TIME}.png')

        ## Visualize UMAP embeddings as 3d
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2])
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        plt.title('UMAP 3D')
        plt.savefig(f'{CFG.REPORTS_PATH}umap_3d_{CURRENT_TIME}.png')
        plt.close()

    return umap_embeddings


def run_kmeans(**kwargs):
    '''
    Runs KMeans on embeddings
    '''
    embeddings = kwargs.get('embeddings')
    n_clusters = kwargs.get('n_clusters', 10)
    random_state = kwargs.get('random_state', 42)
    visualize = kwargs.get('visualize', False)

    ## Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embeddings)

    if visualize:
        ## Visualize KMeans clusters
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=kmeans.labels_)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('KMeans Clusters')
        plt.savefig(f'{CFG.REPORTS_PATH}kmeans_clusters_{CURRENT_TIME}.png')

        ## Visualize KMeans clusters as word clouds
        for i in range(n_clusters):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join([str(x) for x in embeddings[kmeans.labels_ == i, :].mean(axis=0)]))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'KMeans cluster {i+1}')
            plt.savefig(f'{CFG.REPORTS_PATH}kmeans_cluster_{i+1}_{CURRENT_TIME}.png')

        ## Visualize KMeans clusters as bar plots
        for i in range(n_clusters):
            plt.figure(figsize=(10, 5))
            plt.bar(range(len(embeddings[kmeans.labels_ == i, :].mean(axis=0))), embeddings[kmeans.labels_ == i, :].mean(axis=0))
            plt.xlabel('Features')
            plt.ylabel('Value')
            plt.title(f'KMeans cluster {i+1}')
            plt.savefig(f'{CFG.REPORTS_PATH}kmeans_cluster_{i+1}_bar_{CURRENT_TIME}.png')

    return kmeans.labels_


def run_hdbscan(**kwargs):
    '''
    Runs HDBSCAN on embeddings
    '''
    embeddings = kwargs.get('embeddings')
    min_cluster_size = kwargs.get('min_cluster_size', 5)
    min_samples = kwargs.get('min_samples', 5)
    # random_state = kwargs.get('random_state', 42)
    visualize = kwargs.get('visualize', False)
    cleaned_text = kwargs.get('cleaned_text', None)

    ## Run HDBSCAN
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        # random_state=random_state,
        cluster_selection_method='eom'
    )
    hdbscan.fit(embeddings)

    if visualize:
        ## Visualize HDBSCAN clusters
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=hdbscan.labels_)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('HDBSCAN Clusters')
        plt.savefig(f'{CFG.REPORTS_PATH}hdbscan_clusters_{CURRENT_TIME}.png')

        ## Visualize HDBSCAN clusters as word clouds
        for i in range(hdbscan.labels_.max() + 1):
            words = [cleaned_text[j] for j in range(len(hdbscan.labels_)) if hdbscan.labels_[j] == i]
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'HDBSCAN cluster {i+1}')
            plt.savefig(f'{CFG.REPORTS_PATH}hdbscan_cluster_{i+1}_{CURRENT_TIME}.png')

        # ## Visualize HDBSCAN clusters as bar plots
        # for i in range(hdbscan.labels_.max() + 1):
        #     plt.figure(figsize=(10, 5))
        #     plt.bar(range(len(embeddings[hdbscan.labels_ == i, :].mean(axis=0))), embeddings[hdbscan.labels_ == i, :].mean(axis=0))
        #     plt.xlabel('Features')
        #     plt.ylabel('Value')
        #     plt.title(f'HDBSCAN cluster {i+1}')
        #     plt.savefig(f'{CFG.REPORTS_PATH}hdbscan_cluster_{i+1}_bar_{CURRENT_TIME}.png')

    return hdbscan.labels_


def get_purity(**kwargs):
    '''
    Calculates purity of clustering
    '''
    embeddings = kwargs.get('embeddings')
    labels = kwargs.get('labels')

    ## Calculate purity
    purity = 0
    for label in set(labels):
        if label == -1:
            continue
        cluster = embeddings[labels == label]
        cluster_label = Counter([x for x in cluster[:, 0]]).most_common(1)[0][0]
        purity += np.sum(cluster[:, 0] == cluster_label) / len(cluster)

    return purity


def get_silhouette_score(**kwargs):
    '''
    Calculates silhouette score of clustering
    '''
    embeddings = kwargs.get('embeddings')
    labels = kwargs.get('labels')

    ## Calculate silhouette score
    silh_score = silhouette_score(embeddings, labels)

    return silh_score


def plot_wordcloud(data, cluster_id):
        '''
        Generate word cloud for cluster and show top 10 words
        '''
        print(f"Generating word cloud for Cluster {cluster_id}...")

        text = " ".join(data[data["labels"] == cluster_id]["cleaned_pos"].astype(str))

        if len(text) == 0:
            print(f"Skipping Cluster {cluster_id} (empty)")
            return

        wordcloud = WordCloud(width=400, height=400, background_color="white", colormap="viridis").generate(text)

        # Extract top 10 words
        words = text.split()
        word_counts = Counter(words)
        top_10_words = word_counts.most_common(10)

        # Print Top 10 Words
        print(f"Top 10 Words in Cluster {cluster_id}:")
        for word, count in top_10_words:
            print(f"{word}: {count}")

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud for Cluster {cluster_id}", fontsize=14)
        plt.show()
        plt.savefig(f'{CFG.REPORTS_PATH}wordcloud_cluster_{cluster_id}_{CURRENT_TIME}.png')
        plt.close()