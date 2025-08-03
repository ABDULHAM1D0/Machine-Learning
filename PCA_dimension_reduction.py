from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

dataset = load_iris()

X = dataset.data
y = dataset.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure()
for i in range(len(dataset.target_names)):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], label = dataset.target_names[i])
    
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of iris dataset")
plt.legend()

#%%

#----------------------------------- 3D --------------------------------------

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

fig = plt.figure(1, figsize=(8,6))
ax = fig.add_subplot(111, projection = "3d", elev = -150, azim = 110)

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c = y, s = 40)

ax.set_title("First three pca dimension of iris dataset")
ax.set_xlabel("1st eugenvector")
ax.set_ylabel("2st eugenvector")
ax.set_zlabel("3st eugenvector")
