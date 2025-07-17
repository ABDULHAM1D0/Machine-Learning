import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import load_digits

dataset = load_digits()

fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (10, 5),
                        subplot_kw = {"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    ax.imshow(dataset.images[i], cmap = "binary", interpolation = "nearest")
    ax.set_title(dataset.target)
    
plt.show()


X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

classifier = SVC(kernel = "linear", random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))