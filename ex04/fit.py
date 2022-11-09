import numpy as np

def predict_(x, theta):
    try:
        if type(x) != np.ndarray or type(theta) != np.ndarray:
            return None
        if not len(x) or not len(theta):
            return None
        extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
        return extended_x.dot(theta)
    except:
        return None

def gradient(x, y, theta):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray:
            return None
        l = len(x)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        nabla_J = x.T.dot(x.dot(theta) - y) / l
        return nabla_J
    except:
        return None

def fit_(x, y, theta, alpha, max_iter):
    try:
        new_theta = theta.astype(float)
        for i in range(max_iter):
            nabla_J = gradient(x, y, new_theta)
            new_theta -= alpha * nabla_J
        return new_theta
    except:
        return None


x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])
# Example 0:
print(predict_(x, theta))
theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
print(theta2)
print(predict_(x, theta2))