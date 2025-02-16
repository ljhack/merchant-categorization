Merchant Categorization with Clustering & NLP

🚀 Overview

This project categorizes merchants from transactional data using NLP and clustering techniques. It leverages:

FastText embeddings for textual representation

UMAP for dimensionality reduction

HDBSCAN and KMeans for clustering

WordCloud & Matplotlib for visualization

📂 Project Structure

merchant-categorization/
├── src/
│   ├── categorization.py   # Clustering logic (HDBSCAN & KMeans)
│   ├── model_training.py   # Main script for training & evaluation
│   ├── name_filtering.py   # Filters human names from merchants
│   ├── preprocessing.py    # Data cleaning & embeddings generation
│   ├── __init__.py         # Module initialization
├── configs.py              # Configurations (paths, parameters)
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
├── .gitignore              # Ignoring unnecessary files
├── files/                  # Fasttext input, embedding models and vectors
├── data/                   # Raw and processed data (optional)
├── reports/                # Generated reports & visualizations (optional)
└── notebooks/              # Jupyter Notebooks for analysis (optional)

📌 Features

Text Preprocessing: Removes noise and tokenizes merchant names.

Dimensionality Reduction: Uses UMAP for better clustering.

Clustering Algorithms:

HDBSCAN: Finds variable-sized clusters with noise handling.

KMeans: Alternative clustering method for comparison.

WordCloud Analysis: Generates cluster-specific word clouds.

🔧 Installation

Clone the repository and install dependencies:

git clone <repo-url>
cd merchant-categorization
pip install -r requirements.txt

📊 Running the Model

To train and cluster data:

python src/model_training.py

To filter merchant names:

python src/name_filtering.py

To preprocess data and generate embeddings:

python src/preprocessing.py

🔬 Results

UMAP + HDBSCAN Clustering

Best parameter settings found:

n_neighbors = 30

n_components = 5

min_cluster_size = 1500

min_samples = 200

Achieved 2 clusters, but merchant vs. human name differentiation remains a challenge.

High noise (~0.45% of data) was observed.

KMeans Clustering

Will be tested next as an alternative to density-based clustering.

📊 Visualization

Sample word cloud for a merchant cluster:


🏆 Key Takeaways

Filtering out human names is crucial. Frequency-based and embedding-based methods were explored.

HDBSCAN provided robust clustering, but fine-tuning is required to balance cluster size and noise.

Next steps: Testing KMeans and refining merchant filtering.

📈 Future Work

Improve merchant filtering by removing outliers using embeddings.

Optimize clustering hyperparameters for better silhouette score.

Explore unsupervised topic modeling for more granular categorization.

🤝 Contributing

Feel free to fork, improve, and submit a PR.

📧 Questions? Reach out at abhibak10@gmail.com

