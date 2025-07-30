from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers = 4, cluster_std=1.2, random_state=42)


plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.title("Sample Dataset") 

cluster = KMeans(n_clusters = 4)
cluster.fit(X)

labels = cluster.labels_

plt.figure()
plt.scatter(X[:,0],X[:,1], c = labels, cmap = "viridis")

centers = cluster.cluster_centers_
plt.scatter(centers[:,0],centers[:,1], c = "red", cmap = "X")

