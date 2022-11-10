import numpy as np

def add_polynomial_features(x, power):
    try:
        if type(x) != np.ndarray or x.shape[1] != 1 or not isinstance(power, (int, float)):
            return None
        return np.vander(x.reshape(-1), power + 1 , increasing=True)[:,1:]
    except:
        return None

def main_test():
    x = np.arange(1,6).reshape(-1, 1)
    print(add_polynomial_features(x, 3))
    print(add_polynomial_features(x, 6))

if __name__ == "__main__":
    main_test()