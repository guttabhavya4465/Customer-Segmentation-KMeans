import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=200, centers=5, random_state=42)
df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
print("First 5 rows of dataset:\n")
print(df.head())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
plt.figure()
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
print("\nCluster labels added!")
print("\nClustering Evaluation Metrics:")
print("Silhouette Score:", round(silhouette_score(X_scaled, clusters), 3))
print("Davies-Bouldin Index:", round(davies_bouldin_score(X_scaled, clusters), 3))
print("Calinski-Harabasz Score:", round(calinski_harabasz_score(X_scaled, clusters), 3))
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='X', s=200)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("Customer Segmentation using K-Means")
plt.show()
print("\nCluster-wise Mean Values:")
print(df.groupby('Cluster').mean())
df.to_csv("Clustered_Customers.csv", index=False)
print("\nClustered data saved as 'Clustered_Customers.csv'")
