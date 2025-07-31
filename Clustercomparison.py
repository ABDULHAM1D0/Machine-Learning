from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import cluster

number_samp = 1500
circles = datasets.make_circles(n_samples=number_samp, noise=0.05, factor = 0.5)
moons = datasets.make_moons(n_samples=number_samp, noise=0.05)
blobs = datasets.make_blobs(n_samples=number_samp)
no_structure = np.random.rand(number_samp,2), None

clustering_names = ["MiniBatchKmeans", "SpectralClustering", "Ward",
                    "AgglomerativeClustering", "DBSCAN", "Birch"]

color = np.array(["b", "g", "r", "c", "m", "y"])
dataset = [circles, moons, blobs, no_structure]

i = 1
for index, data in enumerate(dataset):
    X, y = data
    StandardScaler().fit_transform(X)
    
    minibatch = cluster.MiniBatchKMeans(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward")
    spectral = cluster.SpectralClustering(n_clusters=2)
    dbscan = cluster.DBSCAN(eps = 0.5)
    average = cluster.AgglomerativeClustering(n_clusters=2, linkage="average")
    birch = cluster.Birch(n_clusters=2)
    
    clusters = [minibatch, ward, spectral, dbscan, average, birch]
    
    for name, algorithm in zip(clustering_names, clusters):
        
        algorithm.fit(X)
        
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)
            
            
        plt.subplot(len(dataset), len(clusters), i)
        if index == 0:
            plt.title(name)
            
        
        plt.scatter(X[:, 0], X[:, 1], color = color[y_pred].tolist(), s=10)
        
        i += 1
        
    