import numpy as np

def simple_predict(x, theta):
    try:
        if type(x) != np.ndarray or type(theta) != np.ndarray:
            return None
        if not len(x) or not len(theta):
            return None
        y_hat = np.zeros((x.shape[0], theta.shape[1]))
        for i in range(x.shape[0]):
            y_hat[i] += theta[0][0]
            for j in range(x.shape[1]):
                y_hat[i] += x[i][j] * theta[j + 1][0]
        return y_hat
    except:
        return None


def main_test():
    x = np.arange(1,13).reshape((4,-1))
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta1))
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta2))
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print(simple_predict(x, theta3))
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(simple_predict(x, theta4))


if __name__ == "__main__":
    main_test()