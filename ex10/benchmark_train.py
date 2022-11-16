import sys
sys.path.append('../')
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex09.data_spliter import data_spliter
from ex08.polynomial_train import add_polynomial_features
from  ex07.polynomial_model import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

def perform_regression(X, Y, theta):
    print("start")
    myLR =  MyLR(theta, alpha = 1e-2, max_iter = 1000000)
    myLR.fit_(X, Y)
    print("end")
    return myLR


def vizualize_models_perf(perfs):
    plt.plot(list(perfs.keys()),list(perfs.values()))
    plt.xlabel("Models")
    plt.ylabel("MSE")
    plt.show()


def save_models(results):
    file = open('models.pickle', 'wb')
    pickle.dump(results, file)
    file.close()
    
def evaluate_models(models, X_test, Y_test):
    best_model = ""
    best_mse = -1
    for elem in models:
        myLR =  models[elem]
        X_weight = add_polynomial_features(X_test[:, 0].reshape(-1, 1), int(elem[1]))
        X_distance = add_polynomial_features(X_test[:, 1].reshape(-1, 1), int(elem[3]))
        X_time = add_polynomial_features(X_test[:, 2].reshape(-1, 1), int(elem[5]))
        enginered_X = np.hstack((X_weight, X_distance, X_time))
        Y_hat = myLR.predict_(enginered_X)
        current_mse = MyLR.mse_(Y_test, Y_hat)
        if best_mse < 0 or current_mse < best_mse:
            best_model = elem
            best_mse = current_mse
    print("best model :", best_model, ", ", best_mse)

def normalize(X):
    mean_X = np.mean(X)
    std_X = np.std(X)
    return (X - mean_X) / std_X 

def engine_features(X):
    poly_x =  add_polynomial_features(X, 4)
    for i in range(4):
        poly_x[:, i] = normalize(poly_x[:, i])
    return poly_x

def regression_engine(data):    
    X = np.array(data[["weight", "prod_distance", "time_delivery"]])
    Y = np.array(data[["target"]])
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.5)
    X_train_weight = engine_features(X_train[:, 0].reshape(-1, 1))
    X_train_distance = engine_features(X_train[:, 1].reshape(-1, 1))
    X_train_time = engine_features(X_train[:, 2].reshape(-1, 1))
    Y_train = normalize(Y_train)
    results = {}
    for w_rank in range(1, 5):
        for d_rank in range(1, 5):
            for t_rank in range(1, 5):
                print(f"w{w_rank}d{d_rank}t{t_rank}")
                X_train_features = np.hstack((X_train_weight[:,:w_rank], X_train_distance[:,:d_rank], X_train_time[:, :t_rank]))
                theta = np.random.rand(X_train_features.shape[1] + 1, 1).reshape(-1, 1)
                results[f"w{w_rank}d{d_rank}t{t_rank}"] = perform_regression(X_train_features, Y_train, theta)
    save_models(results)
    evaluate_models(results, X_test, Y_test)
    return
    

def main(): 
    data = pd.read_csv("space_avocado.csv")
    regression_engine(data)
    return 

if __name__ == "__main__":
    main()