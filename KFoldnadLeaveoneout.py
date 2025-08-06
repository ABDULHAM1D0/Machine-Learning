from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, GridSearchCV

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier()
tree_params = {"max_depth": [3, 5, 7]}


#KFold Grid Search
kf = KFold(n_splits=10)
tree_grid_search_kf = GridSearchCV(tree, tree_params, cv = kf)
tree_grid_search_kf.fit(X_test, y_test)
print("KF best parameters: ", tree_grid_search_kf.best_params_) 
print("KF best accuracy: ", tree_grid_search_kf.best_score_) 


#LOO
loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_params, cv = loo)
tree_grid_search_loo.fit(X_test, y_test)
print("LOO best parameters: ", tree_grid_search_loo.best_params_) 
print("LOO best accuracy: ", tree_grid_search_loo.best_score_) 
