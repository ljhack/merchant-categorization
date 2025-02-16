'''
Model training for merchant categorization

Run this script with the following command
python run model_training.py --reducer=pca --clusterer=hdbscan
'''

## Go to ROOT
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product

import src.preprocessing as prep
import src.categorization as cluster
import configs as CFG

CURRENT_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M')
TEXT_COL = 'pos'


## HDBSCAN clustering
# def run_hdbscan_model(**kwargs):
def run_hdbscan_model():
    '''
    
    '''
    # data_loc = kwargs.get('data_loc')
    # text_col = kwargs.get('text_col', 'pos')


    ## Read data
    data = pd.read_csv(CFG.INPUT_DATA)
    print(f'Number of rows in data: {len(data)}')

    ## Preprocess data
    scaled_embeddings = prep.preprocess_data(data=data, text_col=CFG.TEXT_COL)
    print(f'Shape of embeddings: {scaled_embeddings.shape}')

    ## Create an Excel Writer
    excel_filename = f'{CFG.REPORTS_PATH}merchant_categorization_{CURRENT_TIME}.xlsx'
    writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')

    ## Metadata storage
    metadata = {
        'reducer': [],
        'clusterer': [],
        'n_components': [],
        'n_neighbors': [],
        'min_cluster_size': [],
        'min_samples': [],
        'metric': [],
        'n_clusters': [],
        'labels': [],
        'noise_ratio': [],
        'cluster_distribution_confidence': [],
        'purity': [],
        'silhouette_score': [],
        'time_taken': []
    }

    ## Track best setting
    best_setting = None
    best_noise_ratio = 1.0  # Initialize with worst possible noise % (100%)
    best_cluster_distribution_confidence = 0   # Initialize with worst possible confidence (0%)

    ## Iterate over parameter combinations
    # n_neighbors_list = [20, 50, 100, 500]  
    # n_components_list = [5, 15, 25]
    # min_cluster_size_list = [100, 500, 1000]
    # min_samples_list = [50, 100, 200]

    ## Temp fastest config for initial run
    n_neighbors_list = [15]
    n_components_list = [10]
    min_cluster_size_list = [100]
    min_samples_list = [50]


    for n_neighbours, n_components, min_cluster_size, min_samples in product(n_neighbors_list, n_components_list, min_cluster_size_list, min_samples_list):
        print(f'Running HDBSCAN with n_neighbors={n_neighbours}, n_components={n_components}, min_cluster_size={min_cluster_size}, min_samples={min_samples}')
        run_id = f'{n_neighbours}_{n_components}_{min_cluster_size}_{min_samples}'
        print(f'Run ID: {run_id}')
        start_time = time.time()

        ## Run UMAP
        umap_embeddings = cluster.run_umap(scaled_embeddings=scaled_embeddings, n_components=n_components, n_neighbors=n_neighbours, visualize=True)

        ## Run HDBSCAN
        labels = cluster.run_hdbscan(embeddings=umap_embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples, visualize=True)
        time_taken = time.time() - start_time
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        ## Use umap embeddings and labels to get purity and silhouette score
        purity = cluster.get_purity(umap_embeddings, labels)
        silhouette_score = cluster.get_silhouette_score(umap_embeddings, labels)

        ## Calculate noise ratio
        noise_ratio = np.sum(labels == -1) / len(labels)

        ## Calculate cluster distribution confidence
        ## Exclude noise points
        cluster_distribution = pd.Series(labels).value_counts().sort_index()
        cluster_distribution = cluster_distribution[cluster_distribution.index != -1]

        ## Biggest cluster shouldn't completely over shadow second biggest
        cluster_distribution_confidence = cluster_distribution.iloc[0] / cluster_distribution.iloc[1]
        if cluster_distribution_confidence > 1:
            cluster_distribution_confidence = 1 / cluster_distribution_confidence
        
        ## Print results
        print(f'Number of clusters: {n_clusters}')
        print(f'Noise ratio: {noise_ratio}')
        print(f'Cluster distribution confidence: {cluster_distribution_confidence}')
        print(f'Purity: {purity}')
        print(f'Silhouette score: {silhouette_score}')
        print(f'Time taken: {time_taken} seconds')

        ## Update best setting if necessary
        if noise_ratio < best_noise_ratio or (noise_ratio == best_noise_ratio and cluster_distribution_confidence > best_cluster_distribution_confidence):
            best_setting = {
                'reducer': 'UMAP',
                'clusterer': 'HDBSCAN',
                'n_components': n_components,
                'n_neighbors': n_neighbours,
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'labels': labels,
                'purity': purity,
                'silhouette_score': silhouette_score,
                'noise_ratio': noise_ratio,
                'cluster_distribution_confidence': cluster_distribution_confidence,
                'time_taken': time_taken
            }
            best_noise_ratio = noise_ratio
            best_cluster_distribution_confidence = cluster_distribution_confidence

        ## Store metadata
        metadata['reducer'].append('UMAP')
        metadata['clusterer'].append('HDBSCAN')
        metadata['n_components'].append(n_components)
        metadata['n_neighbors'].append(n_neighbours)
        metadata['min_cluster_size'].append(min_cluster_size)
        metadata['min_samples'].append(min_samples)
        metadata['n_clusters'].append(n_clusters)
        metadata['labels'].append(labels)
        metadata['purity'].append(purity)
        metadata['silhouette_score'].append(silhouette_score)
        metadata['noise_ratio'].append(noise_ratio)
        metadata['time_taken'].append(time_taken)

        ## Save clustering results for further analysis
        cluster_df = pd.DataFrame({
            'merchant_name': data['pos'],
            'labels': labels
        })
        cluster_df.to_excel(writer, sheet_name=f'run_{run_id}', index=False)

    ## Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_excel(writer, sheet_name='metadata', index=False)
    writer.close()
    print(f'Metadata saved to {excel_filename}')

    ## Save best setting
    if best_setting:
        best_setting_df = pd.DataFrame(best_setting)
        best_setting_df.to_excel(f'{CFG.REPORTS_PATH}best_setting_{CURRENT_TIME}.xlsx', index=False)
        print(f'Best setting saved to {CFG.REPORTS_PATH}best_setting_{CURRENT_TIME}.xlsx')
        print(f'Best setting: {best_setting}')

        df_best = pd.read_excel(f'{CFG.REPORTS_PATH}best_setting_{CURRENT_TIME}.xlsx')

        for cluster_id in df_best['labels'][0]:
            cluster_df = data[data['cluster'] == cluster_id]
            cluster_df.to_excel(f'{CFG.REPORTS_PATH}cluster_{cluster_id}_{CURRENT_TIME}.xlsx', index=False)
            print(f'Cluster {cluster_id} saved to {CFG.REPORTS_PATH}cluster_{cluster_id}_{CURRENT_TIME}.xlsx')

            ## Generate word cloud for cluster
            cluster.plot_wordcloud(data, cluster_id)

    print('Done!')


if __name__ == '__main__':
    run_hdbscan_model()
