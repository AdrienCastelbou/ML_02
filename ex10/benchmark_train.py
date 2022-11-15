import sys
sys.path.append('../')
from ex05.mylinearregression import MyLinearRegression as MyLR
from ex09.data_spliter import data_spliter
from ex08.polynomial_train import add_polynomial_features
from  ex07.polynomial_model import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import *
import csv


def perform_regression(X, Y, theta):
    myLR =  MyLR(theta, alpha = 0.2e-10, max_iter = 50)
    myLR.fit_(X, Y)
    return myLR.thetas


def vizualize_models_perf(perfs):
    plt.plot(list(perfs.keys()),list(perfs.values()))
    plt.xlabel("Models")
    plt.ylabel("MSE")
    plt.show()

def save_models_parameters(results):
    max_theta = 0
    entries = []
    for elem in results:
        if results[elem].shape[0] > max_theta:
            max_theta = results[elem].shape[0]
        entry = []
        entry.append(elem)
        for theta in results[elem]:
            entry.append(theta[0])
        entries.append(entry)
    header = ["name"]
    for i in range(max_theta):
        header.append(f"theta{i}")
    with open('models.csv', 'w') as file:
    # 2. Create a CSV writer
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(entries)

    
def evaluate_models(models, X_test, Y_test):
    best_model = ""
    best_mse = 100000000000000
    for elem in models:
        theta = models[elem]
        myLR =  MyLR(theta)
        X_weight = add_polynomial_features(X_test[:, 0].reshape(-1, 1), int(elem[1]))
        X_distance = add_polynomial_features(X_test[:, 1].reshape(-1, 1), int(elem[3]))
        X_time = add_polynomial_features(X_test[:, 2].reshape(-1, 1), int(elem[5]))
        enginered_X = np.hstack((X_weight, X_distance, X_time))
        Y_hat = myLR.predict_(enginered_X)
        current_mse = MyLR.mse_(Y_test, Y_hat)
        if current_mse < best_mse:
            best_model = elem
            best_mse = current_mse
    print("best model :", best_model)

def regression_engine(data):
    X = np.array(data[["weight", "prod_distance", "time_delivery"]])
    Y = np.array(data[["target"]])
    X_train, X_test, Y_train, Y_test = data_spliter(X, Y, 0.5)
    X_train_weight = X_train[:, 0].reshape(-1, 1)
    X_train_distance = X_train[:, 1].reshape(-1, 1)
    X_train_time = X_train[:, 2].reshape(-1, 1)
    X_train_weight = add_polynomial_features(X_train_weight, 4)
    X_train_distance = add_polynomial_features(X_train_distance, 4)
    X_train_time = add_polynomial_features(X_train_time, 4)
    futures = []
    results = {}
    executor = ProcessPoolExecutor()
    for w_rank in range(1, 3):
        for d_rank in range(1, 3):
            for t_rank in range(1, 3):
                X_train_features = np.hstack((X_train_weight[:,:w_rank], X_train_distance[:,:d_rank], X_train_time[:, :t_rank]))
                future = executor.submit(perform_regression, X_train_features, Y_train, np.random.rand(X_train_features.shape[1] + 1, 1))
                futures.append(future)
                results[f"w{w_rank}d{d_rank}t{t_rank}"] = future
    done, not_done = wait(futures, return_when=ALL_COMPLETED)
    for elem in results:
        results[elem] = results[elem].result()
    save_models_parameters(results)
    evaluate_models(results, X_test, Y_test)
    executor.shutdown()
    return
    

def main(): 
    data = pd.read_csv("space_avocado.csv")
    regression_engine(data)
    return 

if __name__ == "__main__":
    main()