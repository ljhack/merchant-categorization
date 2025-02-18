# Merchant Categorization Using NLP and Clustering Techniques

In the financial and e-commerce sectors, accurately categorizing merchants based on transaction data is essential for analytics, fraud detection, and personalized services. This article details a comprehensive approach to merchant categorization, leveraging Natural Language Processing (NLP) and clustering algorithms. We'll walk through the process step-by-step, incorporating code snippets and data examples to illustrate each phase.

## Project Overview

The primary objective is to develop an ML-driven system that classifies merchants from transactional data. The methodology encompasses data preprocessing, embedding generation, dimensionality reduction, clustering, and visualization (and of course, further iterations of the same!).

**Key Components:**

- **Data Preprocessing:** Cleaning and preparing merchant names.
- **Embedding Generation:** Transforming text data into numerical representations using FastText.
- **Dimensionality Reduction:** Utilizing UMAP to project embeddings into a lower-dimensional space.
- **Clustering:** Applying HDBSCAN and KMeans to group similar merchants.
- **Visualization:** Creating word clouds and plots to interpret clustering results.

## 1. Data Preprocessing

Begin by extracting and cleaning merchant names from your transactional dataset. This step ensures that the text data is uniform and free from noise, which is crucial for accurate embedding generation.

### Example Code:

```python
import pandas as pd
import re

# Load your data
data = pd.read_csv('MerchantsData.csv')

# Function to clean merchant names
def clean_merchant_name(text) -> Union[None, str]:
    '''
    Returns only alphabetical part from text
    '''
    response = None
    pattern = r'[a-z]+'
    parts = re.findall(pattern, str(text), re.IGNORECASE)
    ## Keep only words with at least 2 letters
    if parts := [x.lower() for x in parts if len(x) > 1]:
        response = " ".join(parts)
    return response

# Apply cleaning function
data['cleaned_merchant_name'] = data['merchant_name'].apply(clean_merchant_name)
```

### Sample Data Before and After Cleaning:

|            Original Merchant Name            |            Cleaned Merchant Name               |
|----------------------------------------------|------------------------------------------------|
| BAJAJ FINANCE LIMI	                       | bajaj finance limi                             |
| SHUHARITECHVENTURES	                       | shuharitechventures                            |
| Razorpay software pvt ltd	                   | razorpay software pvt ltd                      |
| PESTSOLUTIONS INC	                           | pestsolutions inc                              |
| ABS ABSOLUTE BARBECUES	                   | abs absolute barbecues                         |
| Bansidhar Packaging	                       | bansidhar packaging                            |
| Ing*Play Games Rummy	                       | ing play games rummy                           |
| Axis Focused 25 Fund - REGULAR GROWTH	       | axis focused fund regular growth               |
| PaytmBBPSCOUHybridflow	                   | paytmbbpscouhybridflow                         |
| HDB FINANCIAL SERVICES LTD	               | hdb financial services ltd                     |
| AVENUE SUPERMARTS LTD	                       | avenue supermarts ltd                          |
| Rashi Peripherals Limited	                   | rashi peripherals limited                      |
| AIRTEL PAYMENT BANK LTD	                   | airtel payment bank ltd                        |

## 2. Embedding Generation with FastText

Transform the cleaned merchant names into numerical vectors using FastText embeddings. These embeddings capture semantic similarities between words, which aids in clustering similar merchants together.

### Example Code:

```python
import fasttext

# Train FastText model
model = fasttext.train_unsupervised('cleaned_merchant_names.txt', model='skipgram', dim=100)

# Generate embeddings
data['embedding'] = data['cleaned_merchant_name'].apply(lambda x: model.get_sentence_vector(x))
```

**Note:** Ensure that `'cleaned_merchant_names.txt'` contains one cleaned merchant name per line.

## 3. Dimensionality Reduction with UMAP

To visualize and cluster the high-dimensional embeddings, reduce their dimensionality using UMAP. This technique preserves the local and global structure of the data.

### Example Code:

```python
import umap
import numpy as np

# Stack embeddings into a numpy array
embeddings = np.vstack(data['embedding'].values)

# Apply UMAP
umap_model = umap.UMAP(n_neighbors=30, n_components=2, metric='cosine')
umap_embeddings = umap_model.fit_transform(embeddings)

# Add UMAP embeddings to DataFrame
data['umap_x'] = umap_embeddings[:, 0]
data['umap_y'] = umap_embeddings[:, 1]
```

## 4. Clustering with HDBSCAN

Utilize HDBSCAN to identify clusters of similar merchants. HDBSCAN is effective for clustering data with varying densities and can handle noise by classifying outliers.

### Example Code:

```python
import hdbscan

# Apply HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, metric='euclidean')
data['cluster'] = clusterer.fit_predict(umap_embeddings)
```

### Interpreting Clustering Results:

| Merchant Name     | Cluster Label |
|-------------------|---------------|
| johns grocery     | 0             |
| starbucks coffee  | 1             |
| walmart store     | 2             |
| unknown merchant  | -1            |

A cluster label of `-1` indicates that the merchant was classified as noise or an outlier.

## 5. Visualization

Visualize the clustering results to interpret and validate the groupings. Word clouds can represent the most frequent terms in each cluster, and scatter plots can display the distribution of clusters.

### Example Code for Scatter Plot:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of UMAP embeddings colored by cluster
plt.figure(figsize=(12, 8))
sns.scatterplot(x='umap_x', y='umap_y', hue='cluster', data=data, palette='tab10', legend='full')
plt.title('UMAP Projection of Merchant Embeddings')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Cluster')
plt.show()
```

### Example Code for Word Cloud:

```python
from wordcloud import WordCloud

# Generate word cloud for a specific cluster
cluster_id = 0
cluster_data = data[data['cluster'] == cluster_id]
text = ' '.join(cluster_data['cleaned_merchant_name'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title(f'Word Cloud for Cluster {cluster_id}')
plt.axis('off')
plt.show()
```

## Conclusion

By following this approach, you can effectively categorize merchants using NLP and clustering techniques. This methodology facilitates better data organization, enhances analytical capabilities, and supports strategic decision-making in financial and e-commerce applications.

For a more detailed exploration and access to the complete code, visit the [GitHub repository](https://github.com/Boxxxi/merchant-categorization).

