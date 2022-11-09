import numpy as np

def loss_(y, y_hat):
    if type(y) != np.ndarray or type(y_hat) != np.ndarray:
        return None
    if y.ndim == 1:
        y = y.reshape(y.shape[0], -1)
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(y_hat.shape[0], -1)
    if y.shape[1] != 1 or y_hat.shape[1] != 1:
        return None
    try:
        return float(1 / (2 * y.shape[0]) * (y_hat - y).T.dot(y_hat - y))
    except:
        return None


def main_test():
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    print(loss_(X, Y))
    print(loss_(X, X)) 

if __name__ == "__main__":
    main_test()