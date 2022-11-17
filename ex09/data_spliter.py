import numpy as np

def data_spliter(x, y, proportion):
    try:
        if type(x) != np.ndarray or type(y) != np.ndarray or type(proportion) != float:
            return None
        n_train = int(proportion * x.shape[0])
        n_test = int((1-proportion) * x.shape[0]) + 1
        perm = np.random.permutation(len(x))
        s_x = x[perm]
        s_y = y[perm]
        x_train, y_train = s_x[:n_train], s_y[:n_train]
        x_test, y_test =  s_x[-n_test:], s_y[-n_test:]
        return x_train, x_test, y_train, y_test
    except:
        return None

def main_test():
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    print(x1)
    print(y)
    print('-----')
    print(data_spliter(x1, y, 0.8))
    print(data_spliter(x1, y, 0.5))
    print('-----')
    x2 = np.array([[  1, 42],
                [300, 10],
                [ 59,  1],
                [300, 59],
                [ 10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    print(x2)
    print(y)
    print(data_spliter(x2, y, 0.8))
    print(data_spliter(x2, y, 0.5))


if __name__ == "__main__":
    main_test()