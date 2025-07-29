from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100, 1)
y = 3 + 4 * X + np.random.rand(100, 1)


regressor = LinearRegression()
regressor.fit(X, y)

plt.figure()
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = "red", alpha = 0.7)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()

a1 = regressor.coef_[0][0]
print("a1: ", a1)

a0 = regressor.intercept_[0]
print("a0: ", a0)

for i in range(100):
    y_1 = a1 + a0 * X
    plt.plot(X, y_1, color = "green", alpha = 0.7)
    
    

#%%
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)

X = X[:, np.newaxis, 2]

X_train = X[: -20]
X_test = X[-20 :]


y_train = y[: -20]
y_test = y[-20 :]

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print("mse: ", mse)

r2 = r2_score(y_test, y_pred)
print("r2: ", r2)

plt.scatter(X_test, y_test, c = "black")
plt.plot(X_test, y_pred, color = "blue")


