from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X = 4 * np.random.rand(100, 1)
y = 2 + 3 * X**2 + np.random.rand(100, 1)

poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X)

regressor = LinearRegression()
regressor.fit(X_poly, y)

plt.scatter(X, y, color = "blue")

X_test = np.linspace(0, 4, 100).reshape(-1, 1)
X_test_poly = poly_feat.transform(X_test)
y_pred = regressor.predict(X_test_poly)

plt.plot(X_test, y_pred, color = "red")
plt.xlabel("X")
plt.xlabel("Y")
plt.title("Polynomial Regression")


#%%

dataset = fetch_california_housing()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_poly = poly_feat.fit_transform(X_train)
X_test_poly = poly_feat.transform(X_test)

regressor.fit(X_train_poly, y_train)
y_pred = regressor.predict(X_test_poly)

mse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Polynomial regression: ", mse)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Multivariable regression: ", mse)