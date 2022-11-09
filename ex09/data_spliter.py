import numpy as np

def data_spliter(x, y, proportion):
    try:
        l = x.shape[0]
        p = int(l * proportion)
        perm = np.random.permutation(len(x))
        s_x = x[perm]
        s_y = y[perm]
        x_train, y_train = s_x[:p], s_y[:p]
        x_test, y_test =  s_x[p:, -1], s_y[p:, -1]
        return x_train.reshape(1, -1), x_test, y_train.reshape(1, -1), y_test
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