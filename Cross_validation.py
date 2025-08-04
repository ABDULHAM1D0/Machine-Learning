from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np




dataset = load_iris()
X = dataset.data 
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



dt = DecisionTreeClassifier()
dt_param = {"max_depth": [3, 5, 7],
            "max_leaf_nodes": [None, 5, 10, 20, 30, 40, 50]}

dt_grid_search = GridSearchCV(dt, dt_param)
dt_grid_search.fit(X_train, y_train)
print("DT Grid Search best parameters:", dt_grid_search.best_params_)
print("DT Grid Search best accuracy:", dt_grid_search.best_score_)

for mean_score, params in zip(dt_grid_search.cv_results_["mean_test_score"], dt_grid_search.cv_results_["params"]):
    print(f"Average test score: {mean_score}, Parameters: {params}")
    

number_cv = 3
cv_result = dt_grid_search.cv_results_
for i, params in enumerate((cv_result["params"])):
    print(f'Parameters: {params}')
    
    for j in range(number_cv):
        accuracy = cv_result[f"split{j}_test_score"][i]
        print(f"\tFold {j + 1} - Accuracy: {accuracy}")