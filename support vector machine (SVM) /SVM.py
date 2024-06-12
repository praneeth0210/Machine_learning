import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        
        self.w = np.zeros(n_features)
        self.b = 0

        
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Only use the first two classes (binary classification)
X = X[y != 2]
y = y[y != 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train our custom SVM model
svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=10000)
svm.fit(X_train, y_train)

# Predict the labels for the test set
predictions = svm.predict(X_test)

# Evaluate the model
acc = accuracy(y_test, predictions)
print("SVM accuracy on Iris dataset:", acc)
