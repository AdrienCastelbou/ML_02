import numpy as np

def predict_(x, theta):
    extended_x = np.hstack((np.ones((x.shape[0], 1)), x))
    return extended_x.dot(theta)

def main_test():
    x = np.arange(1,13).reshape((4,-1))
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print(predict_(x, theta1))
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print(predict_(x, theta2))
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print(predict_(x, theta3))
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(predict_(x, theta4))


if __name__ == "__main__":
    main_test()