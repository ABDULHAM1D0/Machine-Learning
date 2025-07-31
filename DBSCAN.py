from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, _ = make_circles(n_samples=1000, noise = 0.08, factor = 0.5, random_state=42)

plt.figure()
plt.scatter(X[:, 0], X[:, 1])

cluster = DBSCAN(eps = 0.15, min_samples=15)
label = cluster.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = label, cmap = "viridis")
plt.title("DBSCAN results")