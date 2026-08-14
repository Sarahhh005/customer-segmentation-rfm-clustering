# Customer Segmentation with RFM Clustering

An unsupervised learning project that segments customers of an online retail store using the **Online Retail II (UCI)** dataset. The workflow cleans transactional data, engineers an **RFM (Recency, Frequency, Monetary)** feature set per customer, and applies multiple clustering algorithms to discover meaningful customer segments.

## Dataset
[Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — transactional data from a UK-based online retailer, containing invoices, product descriptions, quantities, prices, and customer IDs.

## Workflow

1. **Data Cleaning**
   - Removed rows with missing `Customer ID`
   - Removed duplicate transactions
   - Removed outliers in `Quantity` and `Price` using the IQR method

2. **Feature Engineering**
   - Created `TotalPrice` = `Quantity × Price`
   - Extracted `InvoiceYear`, `InvoiceMonth`, `InvoiceDay`, `InvoiceHour` from `InvoiceDate`
   - Built an **RFM dataset**: Frequency, Monetary, and Quantity per customer

3. **Encoding & Scaling**
   - Label-encoded the `Country` column
   - Scaled numeric features with `RobustScaler` (robust to outliers)

4. **Cluster Number Selection**
   - Elbow Method (inertia) and Silhouette Score across k = 2–10

5. **Clustering Models**
   - **K-Means**
   - **DBSCAN** (with epsilon tuning via Silhouette Score)
   - **Agglomerative Clustering**

6. **Evaluation & Visualization**
   - Compared models using Silhouette Score and Davies-Bouldin Index
   - Visualized clusters in 2D using PCA

## Results
A comparison table summarizes Silhouette Score and Davies-Bouldin Score across K-Means and DBSCAN, helping identify the most well-separated and cohesive customer segments.

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (KMeans, DBSCAN, AgglomerativeClustering, PCA, RobustScaler)
- seaborn, matplotlib

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook ML_-_Clustering.ipynb
```

Download the dataset from Kaggle/UCI and update the file path in the notebook accordingly.
