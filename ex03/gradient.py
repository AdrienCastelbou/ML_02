import numpy as np

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

def main_test():
    x = np.array([
    [ -6,  -7,  -9],
    [ 13,  -2,  14],
    [ -7, 14, -1],
    [-8, -4, 6],
    [-5, -9, 6],
    [ 1, -5, 11],
    [9,-11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    theta1 = np.array([0, 3,0.5,-6]).reshape((-1, 1))
    print(gradient(x, y, theta1))
    theta2 = np.array([0, 0,0,0]).reshape((-1, 1))
    print(gradient(x, y, theta2))

if __name__ == "__main__":
    main_test()