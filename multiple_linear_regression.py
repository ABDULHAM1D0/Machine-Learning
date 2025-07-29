from sklearn.linear_model import  LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = np.random.rand(100, 2)
coef = np.array([3, 5])
#y = 0 + np.dot(X, coef)
#y = 5 +1 0 * np.random.rand(100) + np.dot(X, coef)
y = np.random.rand(100) + np.dot(X, coef)

regressor = LinearRegression()
regressor.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")


x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = regressor.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha = 0.3)
plt.title("Multiple Linear Regression")

print("Coefficient: ", regressor.coef_)
print("Intercept: ", regressor.intercept_)


#%%

dataset = load_diabetes()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("rmse: ", rmse)