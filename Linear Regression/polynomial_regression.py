import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)
X = np.array([2 - 3 * random.gauss(0, 1) for _ in range(20)])
y = np.array([x - 2 * (x ** 2) + random.gauss(-3, 3) for x in X])

# Expand to polynomial features manually (degree=2)
X_poly = np.array([[1, x, x**2] for x in X])

# Implement a simple linear regression using numpy
def fit(X, y):
    X_transpose_X = np.dot(X.T, X)
    X_transpose_y = np.dot(X.T, y)
    beta = np.linalg.solve(X_transpose_X, X_transpose_y)
    return beta

# Fit the model
beta = fit(X_poly, y)

# Make predictions
def predict(X, beta):
    return np.dot(X, beta)

X_fit = np.arange(min(X), max(X), 0.1)
X_fit_poly = np.array([[1, x, x**2] for x in X_fit])
y_pred = predict(X_fit_poly, beta)

# Plotting
def plot(X, y, X_fit, y_pred):
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X_fit, y_pred, color='red', label='Polynomial Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression')
    plt.legend()
    plt.show()

plot(X, y, X_fit, y_pred)
