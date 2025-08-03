from sklearn.datasets import fetch_openml, load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
dataset = fetch_openml("mnist_784", version = 1)

X = dataset.data
y = dataset.target.astype(int)


lda = LinearDiscriminantAnalysis(n_components=2)

X_lda = lda.fit_transform(X, y)

plt.figure()
plt.scatter(X_lda[:, 0], X_lda[:, 1], c = y, cmap = "tab10", alpha = 0.6)
plt.title("LDA of Mnist dataset")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label="digits")


#%%
#--------------------------LDA vs PCA---------------------

colors = ["red", "blue", "green"]

dataset = load_iris()
X = dataset.data
y = dataset.target
target_names = dataset.target_names

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y==i, 0], X_pca[y==i, 1], color = color, alpha = 0.8, label = target_name)
    
plt.legend()
plt.title("PCA of Iris dataset")

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y==i, 0], X_lda[y==i, 1], color = color, alpha = 0.8, label = target_name)
    
plt.legend()
plt.title("LDA of Iris dataset")


#%% 

# -----------------------------TSNE ---------------------------
from sklearn.manifold import TSNE
dataset = fetch_openml("mnist_784", version = 1)

X = dataset.data
y = dataset.target.astype(int)

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = y, cmap = "tab10", alpha = 0.6)
plt.title("TSNE of Mnist dataset")
plt.xlabel("TSNE dimension 1")
plt.ylabel("TSNE dimension 2")















