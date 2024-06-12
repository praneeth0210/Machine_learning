import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Logistic_Regression:
    def __init__(self, lr=0.001, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iter):
            linear_model = np.dot(X, self.weight) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict_prob(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predicted = sigmoid(linear_model)
        return y_predicted

    def predict(self, X):
        y_predicted = self.predict_prob(X)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_pred_class)
    
    def accuracy(self, y_pred, y_true):
        return np.sum(y_pred == y_true) / len(y_true)

# Load and preprocess the dataset
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Logistic_Regression(lr=0.01, num_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_prob = model.predict_prob(X_test)
y_pred = model.predict(X_test)

# Print accuracy
print("Accuracy:", model.accuracy(y_pred, y_test))
