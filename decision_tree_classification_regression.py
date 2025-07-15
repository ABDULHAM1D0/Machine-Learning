from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

dataset = load_iris()

X = dataset.data 
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy", accuracy)

conf = confusion_matrix(y_test, y_pred)
print("Confusion matrix: ")
print(conf)

plt.figure(figsize=(15, 10))
plot_tree(classifier, filled = True, feature_names=dataset.feature_names, class_names=list(dataset.target_names))
plt.show()

feature_importance = classifier.feature_importances_

feature_names = dataset.feature_names

feature_importance_sorted = sorted(zip(feature_importance, feature_names), reverse = True)

for importance, feature in feature_importance_sorted:
    print(f"{feature}: {importance}.")
    


#%%
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import warnings

warnings.filterwarnings("ignore")

dataset = load_iris()


np_class = len(dataset.feature_names)
plot_colors = "ryb"

for pairidx, pair in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
    X = dataset.data[:, pair] 
    y = dataset.target
    
    classifier = DecisionTreeClassifier().fit(X, y)
    
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad = 0.5, w_pad = 0.5, pad = 2.5)
    DecisionBoundaryDisplay.from_estimator(classifier,
                                           X,
                                           cmap = plt.cm.RdYlBu,
                                           response_method="predict",
                                           ax=ax,
                                           xlabel=dataset.feature_names[pair[0]],
                                           ylabel=dataset.feature_names[pair[1]])
    
    for i, color in zip(range(np_class), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c = color, label = dataset.target_names[i],
                    cmap = plt.cm.RdYlBu,
                    edgecolors="black" )
     
plt.legend()
    


# %%
from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



dataset = load_diabetes()

X = dataset.data 
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred,)
print("MSE: ", mse)

rmse = np.sqrt(mse)
print("RMSE: ", rmse)


# %%
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(80,1), axis=0)
y = np.sin(X).ravel()
y[::5] += 5 * (0.5 - np.random.rand(16))

#plt.scatter(X, y)


reg_1 = DecisionTreeRegressor(max_depth=2).fit(X, y)
reg_2 = DecisionTreeRegressor(max_depth=8).fit(X, y)


X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = reg_1.predict(X_test)
y_pred_2 = reg_2.predict(X_test)

plt.figure()
plt.plot(X, y, c = "red", label = "data")
plt.scatter(X, y, c = "red", label = "data")
plt.plot(X_test, y_pred_1, color = "blue", label = "Max_depth 2", linewidth = 2)
plt.plot(X_test, y_pred_2, color = "green", label = "Max_depth 10", linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
















