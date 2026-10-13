import matplotlib.pyplot as plt
import numpy as np
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [5, 6],
    [6, 7],
    [7, 8],
])
y = np.array([0, 0, 0, 1, 1, 1])
X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(X.shape[1])
learning_rate = 0.1
epochs = 1000
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
for i in range(epochs):
  z = np.dot(X, theta)
  y_pred = sigmoid(z)
  gradient = np.dot(X.T, (y_pred - y)) / len(y)
  theta = theta - learning_rate * gradient
y_pred_final = sigmoid(np.dot(X, theta))
y_pred_class = (y_pred_final >= 0.5).astype(int)
accuracy = np.mean(y_pred_class == y)
print("Final Theta:", theta)
print("Predictions:", y_pred_class)
print("Accuracy:", accuracy)
plt.scatter(
    X[:, 1], X[:, 2], c=y, cmap="bwr", edgecolors="k", label="Data points"
)
x_values = np.linspace(0, 8, 100)
y_values = -(theta[0] + theta[1] * x_values) / theta[2]
plt.plot(x_values, y_values, color="black", label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression (Manual)")
plt.legend()
plt.show()
