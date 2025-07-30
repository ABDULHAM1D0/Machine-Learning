from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers = 4, cluster_std=1.2, random_state=42)

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Sample dataset")

linkage_methods = ["ward", "single", "average", "complete"]

plt.figure()
for i, linkage_method in enumerate(linkage_methods, 1):
    
    model = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
    label = model.fit_predict(X)
    
    plt.subplot(2, 4, i)
    plt.title(f"{linkage_method.capitalize()} linkage dendrogram")
    dendrogram(linkage(X, method=linkage_method), no_labels=True)
    plt.xlabel("Data points")
    plt.ylabel("Distance")
    
    plt.subplot(2, 4, i + 4)
    plt.scatter(X[:, 0], X[:, 1], c = label, cmap = "viridis")
    plt.title(f"{linkage_method.capitalize()} linkage dendrogram")
    plt.xlabel("X")
    plt.ylabel("Y")
    
