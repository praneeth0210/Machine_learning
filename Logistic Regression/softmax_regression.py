import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.W = None
        self.b = None

    def initialize_parameters(self, dim, num_classes):
        self.W = np.random.randn(dim, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Corrected axis for numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_cost(self, Y, A):
        m = Y.shape[0]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def gradient_descent(self, X, Y):
        m = X.shape[0]
        for i in range(self.num_iterations):
            # Forward propagation
            Z = np.dot(X, self.W) + self.b
            A = self.softmax(Z)
            
            # Compute cost
            cost = self.compute_cost(Y, A)
            
            # Backward propagation
            dZ = A - Y
            dW = np.dot(X.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Update parameters
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db
            
    
    def fit(self, X, Y):
        num_classes = Y.shape[1]
        num_features = X.shape[1]
        self.initialize_parameters(num_features, num_classes)
        self.gradient_descent(X, Y)

    def predict(self, X):
        Z = np.dot(X, self.W) + self.b
        A = self.softmax(Z)
        return np.argmax(A, axis=1)

# Load and preprocess the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# # One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the softmax regression model
model = SoftmaxRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, Y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
true_labels = np.argmax(Y_test, axis=1)

print("Predictions:", predictions)
print("True labels:", true_labels)

# Calculate accuracy
accuracy = np.mean(predictions == true_labels)
print("Accuracy:", accuracy)
