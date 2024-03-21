import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to the feature matrix
X_b = np.c_[np.ones((100, 1)), X]

# Compute the optimal parameters using the normal equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Print the optimal parameters
print("Optimal parameters:")
print(theta_best)

# Plot the original data and the linear regression line
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta_best), 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Artificial Intelligence')
plt.show()
